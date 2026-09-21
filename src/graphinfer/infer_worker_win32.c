#define WIN32_LEAN_AND_MEAN
#define _WIN32_WINNT 0x0602
#include <windows.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

#include "infer_worker.h"
#include "internal.h"
#include "workers/infer_protocol.h"

typedef struct request_writer {
  HANDLE pipe;
  const uint8_t *data;
  size_t size;
  DWORD error;
} request_writer;

static const uint8_t module_anchor = 0;

static uint32_t load_u32(const uint8_t *data) {
  return (uint32_t)data[0] | ((uint32_t)data[1] << 8) | ((uint32_t)data[2] << 16) |
         ((uint32_t)data[3] << 24);
}

static uint64_t load_u64(const uint8_t *data) {
  return (uint64_t)load_u32(data) | ((uint64_t)load_u32(data + 4) << 32);
}

static void store_u32(uint8_t *data, uint32_t value) {
  data[0] = (uint8_t)value;
  data[1] = (uint8_t)(value >> 8);
  data[2] = (uint8_t)(value >> 16);
  data[3] = (uint8_t)(value >> 24);
}

static void store_u64(uint8_t *data, uint64_t value) {
  store_u32(data, (uint32_t)value);
  store_u32(data + 4, (uint32_t)(value >> 32));
}

static bool view_has_nul(npunlock_view view) {
  return view.size != 0 && memchr(view.data, 0, view.size) != NULL;
}

static wchar_t *utf8_to_wide(npunlock_view value) {
  int count;
  wchar_t *wide;
  if (!npunlock_view_is_valid(value) || value.size == 0 || value.size > INT_MAX ||
      view_has_nul(value)) {
    return NULL;
  }
  count = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, (const char *)value.data,
                              (int)value.size, NULL, 0);
  if (count <= 0) {
    return NULL;
  }
  wide = (wchar_t *)malloc(((size_t)count + 1) * sizeof(*wide));
  if (wide == NULL) {
    return NULL;
  }
  if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, (const char *)value.data, (int)value.size,
                          wide, count) != count) {
    free(wide);
    return NULL;
  }
  wide[count] = L'\0';
  return wide;
}

static wchar_t *default_worker_path(void) {
  static const wchar_t worker_name[] = L"npunlock_worker.exe";
  HMODULE module = NULL;
  wchar_t *path;
  DWORD length;
  wchar_t *separator;
  size_t directory_length;
  if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                              GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                          (LPCWSTR)&module_anchor, &module)) {
    return NULL;
  }
  path = (wchar_t *)malloc(32768 * sizeof(*path));
  if (path == NULL) {
    return NULL;
  }
  length = GetModuleFileNameW(module, path, 32768);
  if (length == 0 || length >= 32768) {
    free(path);
    return NULL;
  }
  separator = wcsrchr(path, L'\\');
  if (separator == NULL) {
    separator = wcsrchr(path, L'/');
  }
  if (separator == NULL) {
    free(path);
    return NULL;
  }
  directory_length = (size_t)(separator - path) + 1;
  if (directory_length + sizeof(worker_name) / sizeof(worker_name[0]) > 32768) {
    free(path);
    return NULL;
  }
  memcpy(path + directory_length, worker_name, sizeof(worker_name));
  return path;
}

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
  store_u32(data, NPUNLOCK_INFER_REQUEST_MAGIC);
  store_u32(data + 4, NPUNLOCK_INFER_PROTOCOL_VERSION);
  store_u32(data + 8, driver_index);
  store_u32(data + 12, device_index);
  store_u64(data + 16, graph_blob.size);
  store_u32(data + 24, (uint32_t)input_count);
  store_u32(data + 28, NPUNLOCK_INFER_REQUEST_INPUT_SIZE);
  store_u64(data + 32, input_payload_size);
  cursor = NPUNLOCK_INFER_REQUEST_HEADER_SIZE;
  for (index = 0; index < input_count; ++index) {
    uint8_t *descriptor = data + cursor + index * NPUNLOCK_INFER_REQUEST_INPUT_SIZE;
    store_u32(descriptor, inputs[index].argument_index);
    store_u32(descriptor + 4, (uint32_t)inputs[index].argument_name_utf8.size);
    store_u64(descriptor + 8, inputs[index].data.size);
    store_u64(descriptor + 16, payload_offset);
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

static DWORD WINAPI write_request(void *context) {
  request_writer *writer = (request_writer *)context;
  const uint8_t *cursor = writer->data;
  size_t remaining = writer->size;
  writer->error = ERROR_SUCCESS;
  while (remaining != 0) {
    DWORD chunk = remaining > MAXDWORD ? MAXDWORD : (DWORD)remaining;
    DWORD written = 0;
    if (!WriteFile(writer->pipe, cursor, chunk, &written, NULL) || written == 0) {
      writer->error = GetLastError();
      break;
    }
    cursor += written;
    remaining -= written;
  }
  CloseHandle(writer->pipe);
  writer->pipe = NULL;
  return 0;
}

static bool response_expected_size(const uint8_t *response, size_t response_size,
                                   size_t *expected) {
  uint32_t output_count;
  uint32_t descriptor_size;
  uint64_t payload_size;
  uint64_t diagnostic_size;
  size_t descriptors_size;
  size_t total = NPUNLOCK_INFER_RESPONSE_HEADER_SIZE;
  if (response_size < NPUNLOCK_INFER_RESPONSE_HEADER_SIZE) {
    return false;
  }
  if (load_u32(response) != NPUNLOCK_INFER_RESPONSE_MAGIC ||
      load_u32(response + 4) != NPUNLOCK_INFER_PROTOCOL_VERSION || load_u32(response + 44) != 0) {
    return false;
  }
  output_count = load_u32(response + 36);
  descriptor_size = load_u32(response + 40);
  diagnostic_size = load_u64(response + 48);
  payload_size = load_u64(response + 56);
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

static bool append_response(uint8_t **data, size_t *size, size_t *capacity, HANDLE pipe,
                            DWORD available) {
  const size_t maximum = NPUNLOCK_INFER_RESPONSE_HEADER_SIZE +
                         NPUNLOCK_INFER_MAX_ARGUMENTS * NPUNLOCK_INFER_RESPONSE_OUTPUT_SIZE +
                         NPUNLOCK_INFER_MAX_OUTPUT_SIZE + NPUNLOCK_INFER_MAX_DIAGNOSTIC_SIZE;
  size_t required;
  DWORD received = 0;
  uint8_t *replacement;
  if (available == 0 || !npunlock_checked_add_size(*size, available, &required) ||
      required > maximum) {
    return available == 0;
  }
  if (required > *capacity) {
    size_t next = *capacity == 0 ? 4096 : *capacity;
    while (next < required) {
      if (next > maximum / 2) {
        next = maximum;
        break;
      }
      next *= 2;
    }
    replacement = (uint8_t *)realloc(*data, next);
    if (replacement == NULL) {
      return false;
    }
    *data = replacement;
    *capacity = next;
  }
  if (!ReadFile(pipe, *data + *size, available, &received, NULL) || received == 0) {
    return false;
  }
  *size += received;
  return true;
}

static bool response_is_complete(const uint8_t *response, size_t response_size, bool *complete) {
  size_t expected;
  *complete = false;
  if (response_size < NPUNLOCK_INFER_RESPONSE_HEADER_SIZE) {
    return true;
  }
  if (!response_expected_size(response, response_size, &expected) || response_size > expected) {
    return false;
  }
  *complete = response_size == expected;
  return true;
}

static npunlock_status parse_response(const uint8_t *response, size_t response_size,
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
  if (!response_expected_size(response, response_size, &expected) || expected != response_size) {
    return NPUNLOCK_STATUS_DRIVER_FAILED;
  }
  result->worker_status = load_u32(response + 8);
  result->driver_result = load_u32(response + 12);
  result->selected_driver_index = load_u32(response + 16);
  result->selected_device_index = load_u32(response + 20);
  result->driver_version = load_u32(response + 24);
  result->device_vendor_id = load_u32(response + 28);
  result->device_id = load_u32(response + 32);
  output_count = load_u32(response + 36);
  diagnostic_size = load_u64(response + 48);
  payload_size = load_u64(response + 56);
  descriptors_size = (size_t)output_count * NPUNLOCK_INFER_RESPONSE_OUTPUT_SIZE;
  payload = response + NPUNLOCK_INFER_RESPONSE_HEADER_SIZE + descriptors_size;
  if (output_count != 0) {
    result->outputs = (graphinfer_output *)calloc(output_count, sizeof(*result->outputs));
    if (result->outputs == NULL) {
      return NPUNLOCK_STATUS_OUT_OF_MEMORY;
    }
  }
  result->output_count = output_count;
  for (index = 0; index < output_count; ++index) {
    const uint8_t *descriptor = response + NPUNLOCK_INFER_RESPONSE_HEADER_SIZE +
                                index * NPUNLOCK_INFER_RESPONSE_OUTPUT_SIZE;
    graphinfer_output *output = &result->outputs[index];
    uint32_t name_size = load_u32(descriptor + 12);
    uint32_t dims_count = load_u32(descriptor + 8);
    uint64_t data_size = load_u64(descriptor + 40);
    uint64_t payload_offset = load_u64(descriptor + 48);
    size_t dimension;
    if (load_u32(descriptor + 36) != 0 || load_u64(descriptor + 56) != 0 || dims_count == 0 ||
        dims_count > GRAPHINFER_MAX_DIMS || load_u32(descriptor + 4) != GRAPHINFER_PRECISION_FP16 ||
        payload_offset != cursor || cursor > payload_size || name_size > payload_size - cursor) {
      return NPUNLOCK_STATUS_DRIVER_FAILED;
    }
    output->struct_size = sizeof(*output);
    output->argument_index = load_u32(descriptor);
    output->precision = load_u32(descriptor + 4);
    output->dims_count = dims_count;
    for (dimension = 0; dimension < GRAPHINFER_MAX_DIMS; ++dimension) {
      output->dims[dimension] = load_u32(descriptor + 16 + dimension * 4);
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

static bool configure_job(HANDLE job) {
  JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits;
  memset(&limits, 0, sizeof(limits));
  limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
  return SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits, sizeof(limits)) !=
         FALSE;
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
  memset(result, 0, sizeof(*result));
}

npunlock_status npunlock_run_infer_worker(npunlock_view worker_executable_utf8,
                                          uint32_t driver_index, uint32_t device_index,
                                          npunlock_view graph_blob, const graphinfer_input *inputs,
                                          size_t input_count, uint32_t timeout_ms,
                                          npunlock_infer_worker_result *result) {
  SECURITY_ATTRIBUTES security = {sizeof(security), NULL, TRUE};
  STARTUPINFOW startup = {0};
  PROCESS_INFORMATION process = {0};
  request_writer writer = {0};
  HANDLE input_read = NULL;
  HANDLE input_write = NULL;
  HANDLE response_read = NULL;
  HANDLE response_write = NULL;
  HANDLE null_output = INVALID_HANDLE_VALUE;
  HANDLE job = NULL;
  HANDLE writer_thread = NULL;
  wchar_t *worker_path = NULL;
  wchar_t *command_line = NULL;
  uint8_t *request = NULL;
  size_t request_size = 0;
  uint8_t *response = NULL;
  size_t response_size = 0;
  size_t response_capacity = 0;
  ULONGLONG deadline;
  bool timed_out = false;
  bool process_exited = false;
  bool response_parsed = false;
  DWORD exit_code = 0;
  npunlock_status status;

  if (result == NULL || timeout_ms == 0) {
    return NPUNLOCK_STATUS_INVALID_ARGUMENT;
  }
  memset(result, 0, sizeof(*result));
  status = build_request(driver_index, device_index, graph_blob, inputs, input_count, &request,
                         &request_size);
  if (status != NPUNLOCK_STATUS_OK) {
    goto done;
  }
  worker_path = worker_executable_utf8.size == 0 ? default_worker_path()
                                                 : utf8_to_wide(worker_executable_utf8);
  if (worker_path == NULL) {
    status = NPUNLOCK_STATUS_INVALID_ARGUMENT;
    goto done;
  }
  if (!CreatePipe(&input_read, &input_write, &security, 0) ||
      !CreatePipe(&response_read, &response_write, &security, 0) ||
      !SetHandleInformation(input_write, HANDLE_FLAG_INHERIT, 0) ||
      !SetHandleInformation(response_read, HANDLE_FLAG_INHERIT, 0)) {
    status = NPUNLOCK_STATUS_INTERNAL_ERROR;
    goto done;
  }
  null_output = CreateFileW(L"NUL", GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, &security,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
  job = CreateJobObjectW(NULL, NULL);
  if (null_output == INVALID_HANDLE_VALUE || job == NULL || !configure_job(job)) {
    status = NPUNLOCK_STATUS_INTERNAL_ERROR;
    goto done;
  }
  command_line = (wchar_t *)malloc((wcslen(worker_path) + 80) * sizeof(*command_line));
  if (command_line == NULL) {
    status = NPUNLOCK_STATUS_OUT_OF_MEMORY;
    goto done;
  }
  swprintf_s(command_line, wcslen(worker_path) + 80, L"\"%ls\" infer --response-handle %llu",
             worker_path, (unsigned long long)(uintptr_t)response_write);
  startup.cb = sizeof(startup);
  startup.dwFlags = STARTF_USESTDHANDLES;
  startup.hStdInput = input_read;
  startup.hStdOutput = null_output;
  startup.hStdError = null_output;
  if (!CreateProcessW(worker_path, command_line, NULL, NULL, TRUE,
                      CREATE_NO_WINDOW | CREATE_SUSPENDED, NULL, NULL, &startup, &process)) {
    DWORD error = GetLastError();
    status = error == ERROR_FILE_NOT_FOUND || error == ERROR_PATH_NOT_FOUND
                 ? NPUNLOCK_STATUS_NOT_FOUND
                 : NPUNLOCK_STATUS_INTERNAL_ERROR;
    goto done;
  }
  if (!AssignProcessToJobObject(job, process.hProcess) ||
      ResumeThread(process.hThread) == (DWORD)-1) {
    TerminateProcess(process.hProcess, 125);
    status = NPUNLOCK_STATUS_INTERNAL_ERROR;
    goto done;
  }
  CloseHandle(process.hThread);
  process.hThread = NULL;
  CloseHandle(input_read);
  input_read = NULL;
  CloseHandle(response_write);
  response_write = NULL;
  CloseHandle(null_output);
  null_output = INVALID_HANDLE_VALUE;
  writer.pipe = input_write;
  writer.data = request;
  writer.size = request_size;
  writer_thread = CreateThread(NULL, 0, write_request, &writer, 0, NULL);
  if (writer_thread == NULL) {
    status = NPUNLOCK_STATUS_INTERNAL_ERROR;
    goto terminate;
  }
  input_write = NULL;
  deadline = GetTickCount64() + timeout_ms;
  for (;;) {
    DWORD available = 0;
    if (!PeekNamedPipe(response_read, NULL, 0, NULL, &available, NULL)) {
      if (GetLastError() == ERROR_BROKEN_PIPE) {
        break;
      }
      status = NPUNLOCK_STATUS_IO_ERROR;
      goto terminate;
    }
    if (available != 0 &&
        !append_response(&response, &response_size, &response_capacity, response_read, available)) {
      status = NPUNLOCK_STATUS_DRIVER_FAILED;
      goto terminate;
    }
    if (!response_parsed) {
      bool complete;
      if (!response_is_complete(response, response_size, &complete)) {
        status = NPUNLOCK_STATUS_DRIVER_FAILED;
        goto terminate;
      }
      if (complete) {
        status = parse_response(response, response_size, result);
        response_parsed = true;
        if (status != NPUNLOCK_STATUS_OK) {
          goto terminate;
        }
      }
    }
    if (WaitForSingleObject(process.hProcess, 0) == WAIT_OBJECT_0) {
      process_exited = true;
      if (available == 0) {
        break;
      }
    }
    if (GetTickCount64() >= deadline) {
      timed_out = true;
      status = NPUNLOCK_STATUS_TIMEOUT;
      goto terminate;
    }
    Sleep(1);
  }
  if (!process_exited) {
    DWORD remaining = GetTickCount64() >= deadline ? 0 : (DWORD)(deadline - GetTickCount64());
    if (WaitForSingleObject(process.hProcess, remaining) != WAIT_OBJECT_0) {
      timed_out = true;
      status = NPUNLOCK_STATUS_TIMEOUT;
      goto terminate;
    }
  }
  if (!GetExitCodeProcess(process.hProcess, &exit_code) || exit_code != 0) {
    result->process_exit_code = exit_code;
    status = NPUNLOCK_STATUS_DRIVER_FAILED;
    goto terminate;
  }
  result->process_exit_code = exit_code;
  if (!response_parsed) {
    status = parse_response(response, response_size, result);
  }

terminate:
  if (timed_out || status != NPUNLOCK_STATUS_OK) {
    TerminateJobObject(job, timed_out ? 124 : 125);
  }
  if (process.hProcess != NULL) {
    WaitForSingleObject(process.hProcess, 5000);
  }
  if (writer_thread != NULL) {
    if (WaitForSingleObject(writer_thread, 5000) != WAIT_OBJECT_0) {
      CancelSynchronousIo(writer_thread);
      WaitForSingleObject(writer_thread, 5000);
    }
    if (writer.error != ERROR_SUCCESS && status == NPUNLOCK_STATUS_OK) {
      status = NPUNLOCK_STATUS_IO_ERROR;
    }
  }

done:
  if (writer.pipe != NULL) {
    CloseHandle(writer.pipe);
  }
  if (writer_thread != NULL) {
    CloseHandle(writer_thread);
  }
  if (process.hThread != NULL) {
    CloseHandle(process.hThread);
  }
  if (process.hProcess != NULL) {
    CloseHandle(process.hProcess);
  }
  if (job != NULL) {
    CloseHandle(job);
  }
  if (input_read != NULL) {
    CloseHandle(input_read);
  }
  if (input_write != NULL) {
    CloseHandle(input_write);
  }
  if (response_read != NULL) {
    CloseHandle(response_read);
  }
  if (response_write != NULL) {
    CloseHandle(response_write);
  }
  if (null_output != INVALID_HANDLE_VALUE) {
    CloseHandle(null_output);
  }
  free(request);
  free(response);
  free(worker_path);
  free(command_line);
  if (status != NPUNLOCK_STATUS_OK && status != NPUNLOCK_STATUS_DRIVER_FAILED &&
      status != NPUNLOCK_STATUS_NOT_FOUND && status != NPUNLOCK_STATUS_UNSUPPORTED) {
    npunlock_infer_worker_result_release(result);
  }
  return status;
}
