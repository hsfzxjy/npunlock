#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "infer_worker.h"
#include "internal.h"
#include "worker_process.h"
#include "worker_protocol.h"
#include "workers/infer_protocol.h"

static npunlock_status build_request(uint32_t driver_index, uint32_t device_index,
                                     npunlock_view graph_blob, const graphinfer_input *inputs,
                                     size_t input_count, uint8_t **request, size_t *request_size) {
  size_t descriptors_size;
  size_t input_payload_size = 0;
  size_t total = NPUNLOCK_INFER_REQUEST_HEADER_SIZE;
  size_t payload_offset = 0;
  size_t cursor;
  size_t index;
  uint8_t *data;
  if (!npunlock_checked_mul_size(input_count, NPUNLOCK_INFER_REQUEST_INPUT_SIZE,
                                 &descriptors_size) ||
      !npunlock_checked_add_size(total, descriptors_size, &total) ||
      !npunlock_checked_add_size(total, graph_blob.size, &total)) {
    return NPUNLOCK_STATUS_OVERFLOW;
  }
  for (index = 0; index < input_count; ++index) {
    if (!npunlock_checked_add_size(input_payload_size, inputs[index].argument_name_utf8.size,
                                   &input_payload_size) ||
        !npunlock_checked_add_size(input_payload_size, inputs[index].data.size,
                                   &input_payload_size)) {
      return NPUNLOCK_STATUS_OVERFLOW;
    }
  }
  if (graph_blob.size > NPUNLOCK_INFER_MAX_GRAPH_SIZE ||
      input_payload_size > NPUNLOCK_INFER_MAX_INPUT_SIZE ||
      !npunlock_checked_add_size(total, input_payload_size, &total)) {
    return NPUNLOCK_STATUS_INVALID_ARGUMENT;
  }
  data = (uint8_t *)calloc(1, total);
  if (data == NULL) {
    return NPUNLOCK_STATUS_OUT_OF_MEMORY;
  }
  npunlock_worker_store_u32(data, NPUNLOCK_INFER_REQUEST_MAGIC);
  npunlock_worker_store_u32(data + 4, NPUNLOCK_INFER_PROTOCOL_VERSION);
  npunlock_worker_store_u32(data + 8, driver_index);
  npunlock_worker_store_u32(data + 12, device_index);
  npunlock_worker_store_u64(data + 16, graph_blob.size);
  npunlock_worker_store_u32(data + 24, (uint32_t)input_count);
  npunlock_worker_store_u32(data + 28, NPUNLOCK_INFER_REQUEST_INPUT_SIZE);
  npunlock_worker_store_u64(data + 32, input_payload_size);
  cursor = NPUNLOCK_INFER_REQUEST_HEADER_SIZE;
  for (index = 0; index < input_count; ++index) {
    uint8_t *descriptor = data + cursor + index * NPUNLOCK_INFER_REQUEST_INPUT_SIZE;
    npunlock_worker_store_u32(descriptor, inputs[index].argument_index);
    npunlock_worker_store_u32(descriptor + 4, (uint32_t)inputs[index].argument_name_utf8.size);
    npunlock_worker_store_u64(descriptor + 8, inputs[index].data.size);
    npunlock_worker_store_u64(descriptor + 16, payload_offset);
    payload_offset += inputs[index].argument_name_utf8.size + inputs[index].data.size;
  }
  cursor += descriptors_size;
  memcpy(data + cursor, graph_blob.data, graph_blob.size);
  cursor += graph_blob.size;
  for (index = 0; index < input_count; ++index) {
    if (inputs[index].argument_name_utf8.size != 0) {
      memcpy(data + cursor, inputs[index].argument_name_utf8.data,
             inputs[index].argument_name_utf8.size);
      cursor += inputs[index].argument_name_utf8.size;
    }
    memcpy(data + cursor, inputs[index].data.data, inputs[index].data.size);
    cursor += inputs[index].data.size;
  }
  *request = data;
  *request_size = total;
  return NPUNLOCK_STATUS_OK;
}

static bool response_expected_size(npunlock_view response, size_t *expected) {
  uint32_t output_count;
  uint32_t descriptor_size;
  uint64_t payload_size;
  uint64_t diagnostic_size;
  size_t descriptors_size;
  size_t total = NPUNLOCK_INFER_RESPONSE_HEADER_SIZE;
  if (response.size < NPUNLOCK_INFER_RESPONSE_HEADER_SIZE ||
      npunlock_worker_load_u32(response.data) != NPUNLOCK_INFER_RESPONSE_MAGIC ||
      npunlock_worker_load_u32(response.data + 4) != NPUNLOCK_INFER_PROTOCOL_VERSION ||
      npunlock_worker_load_u32(response.data + 44) != 0) {
    return false;
  }
  output_count = npunlock_worker_load_u32(response.data + 36);
  descriptor_size = npunlock_worker_load_u32(response.data + 40);
  diagnostic_size = npunlock_worker_load_u64(response.data + 48);
  payload_size = npunlock_worker_load_u64(response.data + 56);
  if (output_count > NPUNLOCK_INFER_MAX_ARGUMENTS ||
      descriptor_size != NPUNLOCK_INFER_RESPONSE_OUTPUT_SIZE ||
      diagnostic_size > NPUNLOCK_INFER_MAX_DIAGNOSTIC_SIZE ||
      payload_size > NPUNLOCK_INFER_MAX_OUTPUT_SIZE ||
      !npunlock_checked_mul_size(output_count, descriptor_size, &descriptors_size) ||
      !npunlock_checked_add_size(total, descriptors_size, &total) ||
      !npunlock_checked_add_size(total, (size_t)payload_size, &total) ||
      !npunlock_checked_add_size(total, (size_t)diagnostic_size, &total)) {
    return false;
  }
  *expected = total;
  return true;
}

static npunlock_status parse_response(npunlock_view response,
                                      npunlock_infer_worker_result *result) {
  uint32_t output_count;
  uint64_t diagnostic_size;
  uint64_t payload_size;
  size_t expected;
  size_t descriptors_size;
  const uint8_t *payload;
  size_t cursor = 0;
  size_t index;
  npunlock_status status;
  if (!response_expected_size(response, &expected) || expected != response.size) {
    return NPUNLOCK_STATUS_DRIVER_FAILED;
  }
  result->worker_status = npunlock_worker_load_u32(response.data + 8);
  result->driver_result = npunlock_worker_load_u32(response.data + 12);
  result->selected_driver_index = npunlock_worker_load_u32(response.data + 16);
  result->selected_device_index = npunlock_worker_load_u32(response.data + 20);
  result->driver_version = npunlock_worker_load_u32(response.data + 24);
  result->device_vendor_id = npunlock_worker_load_u32(response.data + 28);
  result->device_id = npunlock_worker_load_u32(response.data + 32);
  output_count = npunlock_worker_load_u32(response.data + 36);
  diagnostic_size = npunlock_worker_load_u64(response.data + 48);
  payload_size = npunlock_worker_load_u64(response.data + 56);
  descriptors_size = (size_t)output_count * NPUNLOCK_INFER_RESPONSE_OUTPUT_SIZE;
  payload = response.data + NPUNLOCK_INFER_RESPONSE_HEADER_SIZE + descriptors_size;
  if (output_count != 0) {
    result->outputs = (graphinfer_output *)calloc(output_count, sizeof(*result->outputs));
    if (result->outputs == NULL) {
      return NPUNLOCK_STATUS_OUT_OF_MEMORY;
    }
  }
  result->output_count = output_count;
  for (index = 0; index < output_count; ++index) {
    const uint8_t *descriptor = response.data + NPUNLOCK_INFER_RESPONSE_HEADER_SIZE +
                                index * NPUNLOCK_INFER_RESPONSE_OUTPUT_SIZE;
    graphinfer_output *output = &result->outputs[index];
    uint32_t name_size = npunlock_worker_load_u32(descriptor + 12);
    uint32_t dims_count = npunlock_worker_load_u32(descriptor + 8);
    uint64_t data_size = npunlock_worker_load_u64(descriptor + 40);
    uint64_t payload_offset = npunlock_worker_load_u64(descriptor + 48);
    size_t dimension;
    if (npunlock_worker_load_u32(descriptor + 36) != 0 ||
        npunlock_worker_load_u64(descriptor + 56) != 0 || dims_count == 0 ||
        dims_count > GRAPHINFER_MAX_DIMS ||
        (npunlock_worker_load_u32(descriptor + 4) != GRAPHINFER_PRECISION_FP16 &&
         npunlock_worker_load_u32(descriptor + 4) != GRAPHINFER_PRECISION_FP32) ||
        payload_offset != cursor || cursor > payload_size || name_size > payload_size - cursor) {
      return NPUNLOCK_STATUS_DRIVER_FAILED;
    }
    output->struct_size = sizeof(*output);
    output->argument_index = npunlock_worker_load_u32(descriptor);
    output->precision = npunlock_worker_load_u32(descriptor + 4);
    output->dims_count = dims_count;
    for (dimension = 0; dimension < GRAPHINFER_MAX_DIMS; ++dimension) {
      output->dims[dimension] = npunlock_worker_load_u32(descriptor + 16 + dimension * 4);
    }
    status = npunlock_buffer_copy((npunlock_view){payload + cursor, name_size},
                                  &output->argument_name_utf8);
    if (status != NPUNLOCK_STATUS_OK) {
      return status;
    }
    cursor += name_size;
    if (cursor > payload_size || data_size == 0 || data_size > payload_size - cursor) {
      return NPUNLOCK_STATUS_DRIVER_FAILED;
    }
    status =
        npunlock_buffer_copy((npunlock_view){payload + cursor, (size_t)data_size}, &output->data);
    if (status != NPUNLOCK_STATUS_OK) {
      return status;
    }
    cursor += (size_t)data_size;
  }
  if (cursor != payload_size) {
    return NPUNLOCK_STATUS_DRIVER_FAILED;
  }
  status =
      npunlock_buffer_copy((npunlock_view){payload + (size_t)payload_size, (size_t)diagnostic_size},
                           &result->diagnostic);
  if (status != NPUNLOCK_STATUS_OK) {
    return status;
  }
  switch (result->worker_status) {
  case NPUNLOCK_INFER_WORKER_OK:
    return NPUNLOCK_STATUS_OK;
  case NPUNLOCK_INFER_WORKER_OUT_OF_MEMORY:
    return NPUNLOCK_STATUS_OUT_OF_MEMORY;
  case NPUNLOCK_INFER_WORKER_LOADER_NOT_FOUND:
  case NPUNLOCK_INFER_WORKER_NPU_NOT_FOUND:
    return NPUNLOCK_STATUS_NOT_FOUND;
  case NPUNLOCK_INFER_WORKER_GRAPH_EXTENSION_MISSING:
  case NPUNLOCK_INFER_WORKER_UNSUPPORTED:
    return NPUNLOCK_STATUS_UNSUPPORTED;
  case NPUNLOCK_INFER_WORKER_BAD_REQUEST:
    return NPUNLOCK_STATUS_INVALID_ARGUMENT;
  default:
    return NPUNLOCK_STATUS_DRIVER_FAILED;
  }
}

void npunlock_infer_worker_result_release(npunlock_infer_worker_result *result) {
  size_t index;
  if (result == NULL) {
    return;
  }
  for (index = 0; index < result->output_count; ++index) {
    npunlock_buffer_release(&result->outputs[index].argument_name_utf8);
    npunlock_buffer_release(&result->outputs[index].data);
  }
  free(result->outputs);
  npunlock_buffer_release(&result->diagnostic);
  npunlock_buffer_release(&result->stdout_log);
  npunlock_buffer_release(&result->stderr_log);
  memset(result, 0, sizeof(*result));
}

npunlock_status npunlock_run_infer_worker(npunlock_view worker_executable_utf8,
                                          uint32_t driver_index, uint32_t device_index,
                                          npunlock_view graph_blob, const graphinfer_input *inputs,
                                          size_t input_count, uint32_t timeout_ms,
                                          npunlock_infer_worker_result *result) {
  npunlock_worker_process_result process = {0};
  uint8_t *request = NULL;
  size_t request_size = 0;
  const size_t maximum_response =
      NPUNLOCK_INFER_RESPONSE_HEADER_SIZE +
      NPUNLOCK_INFER_MAX_ARGUMENTS * NPUNLOCK_INFER_RESPONSE_OUTPUT_SIZE +
      NPUNLOCK_INFER_MAX_OUTPUT_SIZE + NPUNLOCK_INFER_MAX_DIAGNOSTIC_SIZE;
  npunlock_status status;
  if (result == NULL || timeout_ms == 0) {
    return NPUNLOCK_STATUS_INVALID_ARGUMENT;
  }
  memset(result, 0, sizeof(*result));
  status = build_request(driver_index, device_index, graph_blob, inputs, input_count, &request,
                         &request_size);
  if (status == NPUNLOCK_STATUS_OK) {
    status = npunlock_worker_process_run(worker_executable_utf8, L"infer",
                                         (npunlock_view){request, request_size}, maximum_response,
                                         timeout_ms, &process);
  }
  result->process_exit_code = process.process_exit_code;
  result->stdout_log = process.stdout_log;
  memset(&process.stdout_log, 0, sizeof(process.stdout_log));
  result->stderr_log = process.stderr_log;
  memset(&process.stderr_log, 0, sizeof(process.stderr_log));
  if (status == NPUNLOCK_STATUS_OK) {
    status = parse_response((npunlock_view){process.response.data, process.response.size}, result);
  }
  free(request);
  npunlock_worker_process_result_release(&process);
  return status;
}
