#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "internal.h"
#include "movi_stage.h"
#include "worker_process.h"
#include "worker_protocol.h"

static bool view_has_nul(npunlock_view view) {
  return view.size != 0 && memchr(view.data, 0, view.size) != NULL;
}

static bool add_request_size(size_t *total, size_t prefix, size_t payload) {
  size_t next;
  return npunlock_checked_add_size(*total, prefix, &next) &&
         npunlock_checked_add_size(next, payload, total) &&
         *total <= NPUNLOCK_MOVI_MAX_MESSAGE_SIZE;
}

static npunlock_status build_request(npunlock_view dll_path, npunlock_movi_stage stage,
                                     const npunlock_view *arguments, size_t argument_count,
                                     const npunlock_view *inputs, size_t input_count,
                                     uint8_t **request, size_t *request_size) {
  size_t total = NPUNLOCK_MOVI_REQUEST_HEADER_SIZE;
  size_t offset;
  size_t index;
  uint8_t *data;
  if (!npunlock_view_is_valid(dll_path) || dll_path.size == 0 || dll_path.size > UINT32_MAX ||
      view_has_nul(dll_path) || arguments == NULL || argument_count == 0 ||
      argument_count > UINT32_MAX || inputs == NULL || input_count == 0 ||
      input_count > UINT32_MAX) {
    return NPUNLOCK_STATUS_INVALID_ARGUMENT;
  }
  if (!add_request_size(&total, 0, dll_path.size)) {
    return NPUNLOCK_STATUS_OVERFLOW;
  }
  for (index = 0; index < argument_count; ++index) {
    if (!npunlock_view_is_valid(arguments[index]) || arguments[index].size == 0 ||
        arguments[index].size > UINT32_MAX || view_has_nul(arguments[index]) ||
        !add_request_size(&total, 4, arguments[index].size)) {
      return NPUNLOCK_STATUS_INVALID_ARGUMENT;
    }
  }
  for (index = 0; index < input_count; ++index) {
    if (!npunlock_view_is_valid(inputs[index]) ||
        !add_request_size(&total, 8, inputs[index].size)) {
      return NPUNLOCK_STATUS_INVALID_ARGUMENT;
    }
  }
  data = (uint8_t *)malloc(total);
  if (data == NULL) {
    return NPUNLOCK_STATUS_OUT_OF_MEMORY;
  }
  memset(data, 0, NPUNLOCK_MOVI_REQUEST_HEADER_SIZE);
  npunlock_worker_store_u32(data, NPUNLOCK_MOVI_REQUEST_MAGIC);
  npunlock_worker_store_u32(data + 4, NPUNLOCK_MOVI_PROTOCOL_VERSION);
  npunlock_worker_store_u32(data + 8, (uint32_t)stage);
  npunlock_worker_store_u32(data + 12, (uint32_t)argument_count);
  npunlock_worker_store_u32(data + 16, (uint32_t)input_count);
  npunlock_worker_store_u32(data + 20, (uint32_t)dll_path.size);
  offset = NPUNLOCK_MOVI_REQUEST_HEADER_SIZE;
  memcpy(data + offset, dll_path.data, dll_path.size);
  offset += dll_path.size;
  for (index = 0; index < argument_count; ++index) {
    npunlock_worker_store_u32(data + offset, (uint32_t)arguments[index].size);
    offset += 4;
    memcpy(data + offset, arguments[index].data, arguments[index].size);
    offset += arguments[index].size;
  }
  for (index = 0; index < input_count; ++index) {
    npunlock_worker_store_u64(data + offset, inputs[index].size);
    offset += 8;
    if (inputs[index].size != 0) {
      memcpy(data + offset, inputs[index].data, inputs[index].size);
      offset += inputs[index].size;
    }
  }
  *request = data;
  *request_size = total;
  return NPUNLOCK_STATUS_OK;
}

static npunlock_status parse_response(npunlock_view response, npunlock_movi_result *result) {
  uint32_t worker_status;
  uint64_t output_size;
  uint64_t diagnostic_size;
  size_t expected;
  npunlock_status status;
  if (response.size < NPUNLOCK_MOVI_RESPONSE_HEADER_SIZE ||
      npunlock_worker_load_u32(response.data) != NPUNLOCK_MOVI_RESPONSE_MAGIC ||
      npunlock_worker_load_u32(response.data + 4) != NPUNLOCK_MOVI_PROTOCOL_VERSION) {
    return NPUNLOCK_STATUS_TOOL_FAILED;
  }
  worker_status = npunlock_worker_load_u32(response.data + 8);
  result->worker_status = worker_status;
  result->tool_return = (int32_t)npunlock_worker_load_u32(response.data + 12);
  output_size = npunlock_worker_load_u64(response.data + 16);
  diagnostic_size = npunlock_worker_load_u64(response.data + 24);
  if (output_size > NPUNLOCK_MOVI_MAX_RESULT_SIZE ||
      diagnostic_size > NPUNLOCK_MOVI_MAX_RESULT_SIZE ||
      !npunlock_checked_add_size(NPUNLOCK_MOVI_RESPONSE_HEADER_SIZE, (size_t)output_size,
                                 &expected) ||
      !npunlock_checked_add_size(expected, (size_t)diagnostic_size, &expected) ||
      expected != response.size) {
    return NPUNLOCK_STATUS_TOOL_FAILED;
  }
  status = npunlock_buffer_copy(
      (npunlock_view){response.data + NPUNLOCK_MOVI_RESPONSE_HEADER_SIZE, (size_t)output_size},
      &result->output);
  if (status != NPUNLOCK_STATUS_OK) {
    return status;
  }
  status = npunlock_buffer_copy(
      (npunlock_view){response.data + NPUNLOCK_MOVI_RESPONSE_HEADER_SIZE + (size_t)output_size,
                      (size_t)diagnostic_size},
      &result->diagnostic);
  if (status != NPUNLOCK_STATUS_OK) {
    npunlock_buffer_release(&result->output);
    return status;
  }
  if (worker_status == NPUNLOCK_MOVI_WORKER_LOAD_FAILED) {
    return NPUNLOCK_STATUS_NOT_FOUND;
  }
  if (worker_status != NPUNLOCK_MOVI_WORKER_OK || result->tool_return != 0) {
    return NPUNLOCK_STATUS_TOOL_FAILED;
  }
  return NPUNLOCK_STATUS_OK;
}

void npunlock_movi_result_release(npunlock_movi_result *result) {
  if (result == NULL) {
    return;
  }
  npunlock_buffer_release(&result->output);
  npunlock_buffer_release(&result->diagnostic);
  npunlock_buffer_release(&result->stdout_log);
  npunlock_buffer_release(&result->stderr_log);
  memset(result, 0, sizeof(*result));
}

npunlock_status npunlock_run_movi_stage(npunlock_view worker_executable_utf8,
                                        npunlock_view dll_path_utf8, npunlock_movi_stage stage,
                                        const npunlock_view *arguments, size_t argument_count,
                                        const npunlock_view *inputs, size_t input_count,
                                        uint32_t timeout_ms, npunlock_movi_result *result) {
  npunlock_worker_process_result process = {0};
  uint8_t *request = NULL;
  size_t request_size = 0;
  const size_t maximum_response =
      NPUNLOCK_MOVI_RESPONSE_HEADER_SIZE + (size_t)NPUNLOCK_MOVI_MAX_RESULT_SIZE * 2;
  npunlock_status status;
  if (result == NULL || timeout_ms == 0) {
    return NPUNLOCK_STATUS_INVALID_ARGUMENT;
  }
  memset(result, 0, sizeof(*result));
  status = build_request(dll_path_utf8, stage, arguments, argument_count, inputs, input_count,
                         &request, &request_size);
  if (status == NPUNLOCK_STATUS_OK) {
    status = npunlock_worker_process_run(worker_executable_utf8, L"movi",
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
