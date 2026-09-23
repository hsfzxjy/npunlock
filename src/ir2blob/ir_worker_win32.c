#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "internal.h"
#include "ir_worker.h"
#include "worker_process.h"
#include "worker_protocol.h"
#include "workers/ir_protocol.h"

static bool view_has_nul(npunlock_view view) {
  return view.size != 0 && memchr(view.data, 0, view.size) != NULL;
}

static npunlock_status build_request(uint32_t driver_index, uint32_t device_index,
                                     npunlock_view ir_xml, npunlock_view weights,
                                     npunlock_view build_flags, uint8_t **request,
                                     size_t *request_size) {
  size_t total = NPUNLOCK_IR_REQUEST_HEADER_SIZE;
  size_t offset;
  uint8_t *data;
  if (!npunlock_view_is_valid(ir_xml) || ir_xml.size == 0 || !npunlock_view_is_valid(weights) ||
      !npunlock_view_is_valid(build_flags) || build_flags.size > UINT32_MAX ||
      view_has_nul(build_flags) || !npunlock_checked_add_size(total, ir_xml.size, &total) ||
      !npunlock_checked_add_size(total, weights.size, &total) ||
      !npunlock_checked_add_size(total, build_flags.size, &total) ||
      total > NPUNLOCK_IR_MAX_MESSAGE_SIZE) {
    return NPUNLOCK_STATUS_INVALID_ARGUMENT;
  }
  data = (uint8_t *)calloc(1, total);
  if (data == NULL) {
    return NPUNLOCK_STATUS_OUT_OF_MEMORY;
  }
  npunlock_worker_store_u32(data, NPUNLOCK_IR_REQUEST_MAGIC);
  npunlock_worker_store_u32(data + 4, NPUNLOCK_IR_PROTOCOL_VERSION);
  npunlock_worker_store_u32(data + 8, driver_index);
  npunlock_worker_store_u32(data + 12, device_index);
  npunlock_worker_store_u64(data + 16, ir_xml.size);
  npunlock_worker_store_u64(data + 24, weights.size);
  npunlock_worker_store_u32(data + 32, (uint32_t)build_flags.size);
  offset = NPUNLOCK_IR_REQUEST_HEADER_SIZE;
  memcpy(data + offset, ir_xml.data, ir_xml.size);
  offset += ir_xml.size;
  if (weights.size != 0) {
    memcpy(data + offset, weights.data, weights.size);
    offset += weights.size;
  }
  if (build_flags.size != 0) {
    memcpy(data + offset, build_flags.data, build_flags.size);
  }
  *request = data;
  *request_size = total;
  return NPUNLOCK_STATUS_OK;
}

static npunlock_status parse_response(npunlock_view response, npunlock_ir_worker_result *result) {
  uint64_t blob_size;
  uint64_t diagnostic_size;
  size_t expected;
  npunlock_status status;
  if (response.size < NPUNLOCK_IR_RESPONSE_HEADER_SIZE ||
      npunlock_worker_load_u32(response.data) != NPUNLOCK_IR_RESPONSE_MAGIC ||
      npunlock_worker_load_u32(response.data + 4) != NPUNLOCK_IR_PROTOCOL_VERSION) {
    return NPUNLOCK_STATUS_DRIVER_FAILED;
  }
  result->worker_status = npunlock_worker_load_u32(response.data + 8);
  result->driver_result = npunlock_worker_load_u32(response.data + 12);
  result->selected_driver_index = npunlock_worker_load_u32(response.data + 16);
  result->selected_device_index = npunlock_worker_load_u32(response.data + 20);
  result->graph_extension_version = npunlock_worker_load_u32(response.data + 24);
  result->compiler_version_major = (uint16_t)npunlock_worker_load_u32(response.data + 28);
  result->compiler_version_minor = (uint16_t)(npunlock_worker_load_u32(response.data + 28) >> 16);
  result->max_opset_version = npunlock_worker_load_u32(response.data + 32);
  result->driver_version = npunlock_worker_load_u32(response.data + 36);
  result->device_vendor_id = npunlock_worker_load_u32(response.data + 40);
  result->device_id = npunlock_worker_load_u32(response.data + 44);
  result->elf_version_major = npunlock_worker_load_u32(response.data + 48);
  result->elf_version_minor = npunlock_worker_load_u32(response.data + 52);
  result->elf_version_patch = npunlock_worker_load_u32(response.data + 56);
  result->runtime_version_major = npunlock_worker_load_u32(response.data + 60);
  result->runtime_version_minor = npunlock_worker_load_u32(response.data + 64);
  result->runtime_version_patch = npunlock_worker_load_u32(response.data + 68);
  blob_size = npunlock_worker_load_u64(response.data + 72);
  diagnostic_size = npunlock_worker_load_u64(response.data + 80);
  if (blob_size > NPUNLOCK_IR_MAX_RESULT_SIZE ||
      diagnostic_size > NPUNLOCK_IR_MAX_DIAGNOSTIC_SIZE ||
      !npunlock_checked_add_size(NPUNLOCK_IR_RESPONSE_HEADER_SIZE, (size_t)blob_size, &expected) ||
      !npunlock_checked_add_size(expected, (size_t)diagnostic_size, &expected) ||
      expected != response.size) {
    return NPUNLOCK_STATUS_DRIVER_FAILED;
  }
  status = npunlock_buffer_copy(
      (npunlock_view){response.data + NPUNLOCK_IR_RESPONSE_HEADER_SIZE, (size_t)blob_size},
      &result->graph_blob);
  if (status != NPUNLOCK_STATUS_OK) {
    return status;
  }
  status = npunlock_buffer_copy(
      (npunlock_view){response.data + NPUNLOCK_IR_RESPONSE_HEADER_SIZE + (size_t)blob_size,
                      (size_t)diagnostic_size},
      &result->diagnostic);
  if (status != NPUNLOCK_STATUS_OK) {
    npunlock_buffer_release(&result->graph_blob);
    return status;
  }
  switch (result->worker_status) {
  case NPUNLOCK_IR_WORKER_OK:
    return NPUNLOCK_STATUS_OK;
  case NPUNLOCK_IR_WORKER_OUT_OF_MEMORY:
    return NPUNLOCK_STATUS_OUT_OF_MEMORY;
  case NPUNLOCK_IR_WORKER_LOADER_NOT_FOUND:
  case NPUNLOCK_IR_WORKER_NPU_NOT_FOUND:
    return NPUNLOCK_STATUS_NOT_FOUND;
  case NPUNLOCK_IR_WORKER_GRAPH_EXTENSION_MISSING:
  case NPUNLOCK_IR_WORKER_UNSUPPORTED:
    return NPUNLOCK_STATUS_UNSUPPORTED;
  case NPUNLOCK_IR_WORKER_BAD_REQUEST:
    return NPUNLOCK_STATUS_INVALID_ARGUMENT;
  default:
    return NPUNLOCK_STATUS_DRIVER_FAILED;
  }
}

void npunlock_ir_worker_result_release(npunlock_ir_worker_result *result) {
  if (result == NULL) {
    return;
  }
  npunlock_buffer_release(&result->graph_blob);
  npunlock_buffer_release(&result->diagnostic);
  npunlock_buffer_release(&result->stdout_log);
  npunlock_buffer_release(&result->stderr_log);
  memset(result, 0, sizeof(*result));
}

npunlock_status npunlock_run_ir_worker(npunlock_view worker_executable_utf8, uint32_t driver_index,
                                       uint32_t device_index, npunlock_view ir_xml,
                                       npunlock_view weights, npunlock_view build_flags,
                                       uint32_t timeout_ms, npunlock_ir_worker_result *result) {
  npunlock_worker_process_result process = {0};
  uint8_t *request = NULL;
  size_t request_size = 0;
  const size_t maximum_response = NPUNLOCK_IR_RESPONSE_HEADER_SIZE +
                                  (size_t)NPUNLOCK_IR_MAX_RESULT_SIZE +
                                  NPUNLOCK_IR_MAX_DIAGNOSTIC_SIZE;
  npunlock_status status;
  if (result == NULL || timeout_ms == 0) {
    return NPUNLOCK_STATUS_INVALID_ARGUMENT;
  }
  memset(result, 0, sizeof(*result));
  status = build_request(driver_index, device_index, ir_xml, weights, build_flags, &request,
                         &request_size);
  if (status == NPUNLOCK_STATUS_OK) {
    status = npunlock_worker_process_run(worker_executable_utf8, "ir",
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
