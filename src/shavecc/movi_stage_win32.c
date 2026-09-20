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
#include "movi_stage.h"

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
  static const wchar_t worker_name[] = L"npunlock_movi_worker.exe";
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
  store_u32(data, NPUNLOCK_MOVI_REQUEST_MAGIC);
  store_u32(data + 4, NPUNLOCK_MOVI_PROTOCOL_VERSION);
  store_u32(data + 8, (uint32_t)stage);
  store_u32(data + 12, (uint32_t)argument_count);
  store_u32(data + 16, (uint32_t)input_count);
  store_u32(data + 20, (uint32_t)dll_path.size);
  offset = NPUNLOCK_MOVI_REQUEST_HEADER_SIZE;
  memcpy(data + offset, dll_path.data, dll_path.size);
  offset += dll_path.size;
  for (index = 0; index < argument_count; ++index) {
    store_u32(data + offset, (uint32_t)arguments[index].size);
    offset += 4;
    memcpy(data + offset, arguments[index].data, arguments[index].size);
    offset += arguments[index].size;
  }
  for (index = 0; index < input_count; ++index) {
    store_u64(data + offset, inputs[index].size);
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
  size_t maximum = NPUNLOCK_MOVI_RESPONSE_HEADER_SIZE + (size_t)NPUNLOCK_MOVI_MAX_RESULT_SIZE * 2;
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

static npunlock_status parse_response(const uint8_t *response, size_t response_size,
                                      npunlock_movi_result *result) {
  uint32_t worker_status;
  uint64_t output_size;
  uint64_t diagnostic_size;
  size_t expected;
  npunlock_status status;
  if (response_size < NPUNLOCK_MOVI_RESPONSE_HEADER_SIZE ||
      load_u32(response) != NPUNLOCK_MOVI_RESPONSE_MAGIC ||
      load_u32(response + 4) != NPUNLOCK_MOVI_PROTOCOL_VERSION) {
    return NPUNLOCK_STATUS_TOOL_FAILED;
  }
  worker_status = load_u32(response + 8);
  result->worker_status = worker_status;
  result->tool_return = (int32_t)load_u32(response + 12);
  output_size = load_u64(response + 16);
  diagnostic_size = load_u64(response + 24);
  if (output_size > NPUNLOCK_MOVI_MAX_RESULT_SIZE ||
      diagnostic_size > NPUNLOCK_MOVI_MAX_RESULT_SIZE ||
      !npunlock_checked_add_size(NPUNLOCK_MOVI_RESPONSE_HEADER_SIZE, (size_t)output_size,
                                 &expected) ||
      !npunlock_checked_add_size(expected, (size_t)diagnostic_size, &expected) ||
      expected != response_size) {
    return NPUNLOCK_STATUS_TOOL_FAILED;
  }
  status = npunlock_buffer_copy(
      (npunlock_view){response + NPUNLOCK_MOVI_RESPONSE_HEADER_SIZE, (size_t)output_size},
      &result->output);
  if (status != NPUNLOCK_STATUS_OK) {
    return status;
  }
  status = npunlock_buffer_copy(
      (npunlock_view){response + NPUNLOCK_MOVI_RESPONSE_HEADER_SIZE + (size_t)output_size,
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

static bool response_is_complete(const uint8_t *response, size_t response_size, bool *complete) {
  uint64_t output_size;
  uint64_t diagnostic_size;
  size_t expected;
  *complete = false;
  if (response_size < NPUNLOCK_MOVI_RESPONSE_HEADER_SIZE) {
    return true;
  }
  if (load_u32(response) != NPUNLOCK_MOVI_RESPONSE_MAGIC ||
      load_u32(response + 4) != NPUNLOCK_MOVI_PROTOCOL_VERSION) {
    return false;
  }
  output_size = load_u64(response + 16);
  diagnostic_size = load_u64(response + 24);
  if (output_size > NPUNLOCK_MOVI_MAX_RESULT_SIZE ||
      diagnostic_size > NPUNLOCK_MOVI_MAX_RESULT_SIZE ||
      !npunlock_checked_add_size(NPUNLOCK_MOVI_RESPONSE_HEADER_SIZE, (size_t)output_size,
                                 &expected) ||
      !npunlock_checked_add_size(expected, (size_t)diagnostic_size, &expected) ||
      response_size > expected) {
    return false;
  }
  *complete = response_size == expected;
  return true;
}

static bool configure_job(HANDLE job) {
  JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits;
  memset(&limits, 0, sizeof(limits));
  limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
  return SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits, sizeof(limits)) !=
         FALSE;
}

void npunlock_movi_result_release(npunlock_movi_result *result) {
  if (result == NULL) {
    return;
  }
  npunlock_buffer_release(&result->output);
  npunlock_buffer_release(&result->diagnostic);
  memset(result, 0, sizeof(*result));
}

npunlock_status npunlock_run_movi_stage(npunlock_view worker_executable_utf8,
                                        npunlock_view dll_path_utf8, npunlock_movi_stage stage,
                                        const npunlock_view *arguments, size_t argument_count,
                                        const npunlock_view *inputs, size_t input_count,
                                        uint32_t timeout_ms, npunlock_movi_result *result) {
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

  if (result == NULL || timeout_ms == 0 ||
      (stage != NPUNLOCK_MOVI_STAGE_COMPILE && stage != NPUNLOCK_MOVI_STAGE_ASSEMBLE &&
       stage != NPUNLOCK_MOVI_STAGE_LINK)) {
    return NPUNLOCK_STATUS_INVALID_ARGUMENT;
  }
  memset(result, 0, sizeof(*result));
  memset(&startup, 0, sizeof(startup));
  memset(&process, 0, sizeof(process));
  memset(&writer, 0, sizeof(writer));
  status = build_request(dll_path_utf8, stage, arguments, argument_count, inputs, input_count,
                         &request, &request_size);
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
  startup.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
  startup.hStdError = GetStdHandle(STD_ERROR_HANDLE);
  if (!CreateProcessW(worker_path, command_line, NULL, NULL, TRUE,
                      CREATE_NO_WINDOW | CREATE_SUSPENDED, NULL, NULL, &startup, &process)) {
    status = GetLastError() == ERROR_FILE_NOT_FOUND ? NPUNLOCK_STATUS_NOT_FOUND
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
      status = NPUNLOCK_STATUS_TOOL_FAILED;
      goto terminate;
    }
    if (!response_parsed) {
      bool response_complete;
      if (!response_is_complete(response, response_size, &response_complete)) {
        status = NPUNLOCK_STATUS_TOOL_FAILED;
        goto terminate;
      }
      if (response_complete) {
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
    status = NPUNLOCK_STATUS_TOOL_FAILED;
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
  if (status != NPUNLOCK_STATUS_OK && status != NPUNLOCK_STATUS_TOOL_FAILED &&
      status != NPUNLOCK_STATUS_NOT_FOUND) {
    npunlock_movi_result_release(result);
  }
  return status;
}
