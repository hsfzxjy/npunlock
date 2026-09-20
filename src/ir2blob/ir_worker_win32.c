#define WIN32_LEAN_AND_MEAN
#define _WIN32_WINNT 0x0602
#include <windows.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "internal.h"
#include "ir_worker.h"
#include "workers/ir_protocol.h"

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
  static const wchar_t worker_name[] = L"npunlock_ir_worker.exe";
  HMODULE module = NULL;
  wchar_t *path;
  DWORD length;
  wchar_t *separator;
  size_t directory_length;
  size_t total;
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
  total = directory_length + sizeof(worker_name) / sizeof(worker_name[0]);
  if (total > 32768) {
    free(path);
    return NULL;
  }
  memcpy(path + directory_length, worker_name, sizeof(worker_name));
  return path;
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
  store_u32(data, NPUNLOCK_IR_REQUEST_MAGIC);
  store_u32(data + 4, NPUNLOCK_IR_PROTOCOL_VERSION);
  store_u32(data + 8, driver_index);
  store_u32(data + 12, device_index);
  store_u64(data + 16, ir_xml.size);
  store_u64(data + 24, weights.size);
  store_u32(data + 32, (uint32_t)build_flags.size);
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

static bool append_response(uint8_t **data, size_t *size, size_t *capacity, HANDLE pipe,
                            DWORD available) {
  size_t maximum = NPUNLOCK_IR_RESPONSE_HEADER_SIZE + (size_t)NPUNLOCK_IR_MAX_RESULT_SIZE +
                   NPUNLOCK_IR_MAX_DIAGNOSTIC_SIZE;
  DWORD received = 0;
  size_t required;
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
  uint64_t blob_size;
  uint64_t diagnostic_size;
  size_t expected;
  *complete = false;
  if (response_size < NPUNLOCK_IR_RESPONSE_HEADER_SIZE) {
    return true;
  }
  if (load_u32(response) != NPUNLOCK_IR_RESPONSE_MAGIC ||
      load_u32(response + 4) != NPUNLOCK_IR_PROTOCOL_VERSION) {
    return false;
  }
  blob_size = load_u64(response + 72);
  diagnostic_size = load_u64(response + 80);
  if (blob_size > NPUNLOCK_IR_MAX_RESULT_SIZE ||
      diagnostic_size > NPUNLOCK_IR_MAX_DIAGNOSTIC_SIZE ||
      !npunlock_checked_add_size(NPUNLOCK_IR_RESPONSE_HEADER_SIZE, (size_t)blob_size, &expected) ||
      !npunlock_checked_add_size(expected, (size_t)diagnostic_size, &expected) ||
      response_size > expected) {
    return false;
  }
  *complete = response_size == expected;
  return true;
}

static npunlock_status parse_response(const uint8_t *response, size_t response_size,
                                      npunlock_ir_worker_result *result) {
  uint64_t blob_size;
  uint64_t diagnostic_size;
  size_t expected;
  npunlock_status status;
  if (response_size < NPUNLOCK_IR_RESPONSE_HEADER_SIZE ||
      load_u32(response) != NPUNLOCK_IR_RESPONSE_MAGIC ||
      load_u32(response + 4) != NPUNLOCK_IR_PROTOCOL_VERSION) {
    return NPUNLOCK_STATUS_DRIVER_FAILED;
  }
  result->worker_status = load_u32(response + 8);
  result->driver_result = load_u32(response + 12);
  result->selected_driver_index = load_u32(response + 16);
  result->selected_device_index = load_u32(response + 20);
  result->graph_extension_version = load_u32(response + 24);
  result->compiler_version_major = (uint16_t)load_u32(response + 28);
  result->compiler_version_minor = (uint16_t)(load_u32(response + 28) >> 16);
  result->max_opset_version = load_u32(response + 32);
  result->driver_version = load_u32(response + 36);
  result->device_vendor_id = load_u32(response + 40);
  result->device_id = load_u32(response + 44);
  result->elf_version_major = load_u32(response + 48);
  result->elf_version_minor = load_u32(response + 52);
  result->elf_version_patch = load_u32(response + 56);
  result->runtime_version_major = load_u32(response + 60);
  result->runtime_version_minor = load_u32(response + 64);
  result->runtime_version_patch = load_u32(response + 68);
  blob_size = load_u64(response + 72);
  diagnostic_size = load_u64(response + 80);
  if (blob_size > NPUNLOCK_IR_MAX_RESULT_SIZE ||
      diagnostic_size > NPUNLOCK_IR_MAX_DIAGNOSTIC_SIZE ||
      !npunlock_checked_add_size(NPUNLOCK_IR_RESPONSE_HEADER_SIZE, (size_t)blob_size, &expected) ||
      !npunlock_checked_add_size(expected, (size_t)diagnostic_size, &expected) ||
      expected != response_size) {
    return NPUNLOCK_STATUS_DRIVER_FAILED;
  }
  status = npunlock_buffer_copy(
      (npunlock_view){response + NPUNLOCK_IR_RESPONSE_HEADER_SIZE, (size_t)blob_size},
      &result->graph_blob);
  if (status != NPUNLOCK_STATUS_OK) {
    return status;
  }
  status = npunlock_buffer_copy(
      (npunlock_view){response + NPUNLOCK_IR_RESPONSE_HEADER_SIZE + (size_t)blob_size,
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

static bool configure_job(HANDLE job) {
  JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits;
  memset(&limits, 0, sizeof(limits));
  limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
  return SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits, sizeof(limits)) !=
         FALSE;
}

void npunlock_ir_worker_result_release(npunlock_ir_worker_result *result) {
  if (result == NULL) {
    return;
  }
  npunlock_buffer_release(&result->graph_blob);
  npunlock_buffer_release(&result->diagnostic);
  memset(result, 0, sizeof(*result));
}

npunlock_status npunlock_run_ir_worker(npunlock_view worker_executable_utf8, uint32_t driver_index,
                                       uint32_t device_index, npunlock_view ir_xml,
                                       npunlock_view weights, npunlock_view build_flags,
                                       uint32_t timeout_ms, npunlock_ir_worker_result *result) {
  SECURITY_ATTRIBUTES security = {sizeof(security), NULL, TRUE};
  STARTUPINFOW startup;
  PROCESS_INFORMATION process;
  HANDLE input_read = NULL;
  HANDLE input_write = NULL;
  HANDLE response_read = NULL;
  HANDLE response_write = NULL;
  HANDLE null_output = INVALID_HANDLE_VALUE;
  HANDLE job = NULL;
  HANDLE writer_thread = NULL;
  request_writer writer;
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
  memset(&startup, 0, sizeof(startup));
  memset(&process, 0, sizeof(process));
  memset(&writer, 0, sizeof(writer));
  status = build_request(driver_index, device_index, ir_xml, weights, build_flags, &request,
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
  command_line = (wchar_t *)malloc((wcslen(worker_path) + 64) * sizeof(*command_line));
  if (command_line == NULL) {
    status = NPUNLOCK_STATUS_OUT_OF_MEMORY;
    goto done;
  }
  swprintf_s(command_line, wcslen(worker_path) + 64, L"\"%ls\" --response-handle %llu", worker_path,
             (unsigned long long)(uintptr_t)response_write);
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
    npunlock_ir_worker_result_release(result);
  }
  return status;
}
