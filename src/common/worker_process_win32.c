#define WIN32_LEAN_AND_MEAN
#define _WIN32_WINNT 0x0602
#include <windows.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

#include "internal.h"
#include "worker_process.h"

#define NPUNLOCK_WORKER_MAX_LOG_SIZE (4u * 1024u * 1024u)

typedef struct request_writer {
  HANDLE pipe;
  const uint8_t *data;
  size_t size;
  DWORD error;
} request_writer;

typedef struct byte_stream {
  uint8_t *data;
  size_t size;
  size_t capacity;
  size_t maximum;
  bool closed;
} byte_stream;

static const uint8_t module_anchor = 0;

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

static bool configure_job(HANDLE job) {
  JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits;
  memset(&limits, 0, sizeof(limits));
  limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
  return SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits, sizeof(limits)) !=
         FALSE;
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

static npunlock_status reserve_stream(byte_stream *stream, size_t additional) {
  size_t required;
  size_t next;
  uint8_t *replacement;
  if (!npunlock_checked_add_size(stream->size, additional, &required) ||
      required > stream->maximum) {
    return NPUNLOCK_STATUS_OVERFLOW;
  }
  if (required <= stream->capacity) {
    return NPUNLOCK_STATUS_OK;
  }
  next = stream->capacity == 0 ? 4096 : stream->capacity;
  while (next < required) {
    if (next > stream->maximum / 2) {
      next = stream->maximum;
      break;
    }
    next *= 2;
  }
  replacement = (uint8_t *)realloc(stream->data, next);
  if (replacement == NULL) {
    return NPUNLOCK_STATUS_OUT_OF_MEMORY;
  }
  stream->data = replacement;
  stream->capacity = next;
  return NPUNLOCK_STATUS_OK;
}

static npunlock_status drain_pipe(HANDLE pipe, byte_stream *stream) {
  DWORD available = 0;
  DWORD received;
  npunlock_status status;
  if (stream->closed) {
    return NPUNLOCK_STATUS_OK;
  }
  if (!PeekNamedPipe(pipe, NULL, 0, NULL, &available, NULL)) {
    if (GetLastError() == ERROR_BROKEN_PIPE) {
      stream->closed = true;
      return NPUNLOCK_STATUS_OK;
    }
    return NPUNLOCK_STATUS_IO_ERROR;
  }
  if (available == 0) {
    return NPUNLOCK_STATUS_OK;
  }
  status = reserve_stream(stream, available);
  if (status != NPUNLOCK_STATUS_OK) {
    return status;
  }
  received = 0;
  if (!ReadFile(pipe, stream->data + stream->size, available, &received, NULL) || received == 0) {
    return GetLastError() == ERROR_BROKEN_PIPE ? NPUNLOCK_STATUS_OK : NPUNLOCK_STATUS_IO_ERROR;
  }
  stream->size += received;
  return NPUNLOCK_STATUS_OK;
}

static void drain_after_exit(HANDLE pipe, byte_stream *stream) {
  while (!stream->closed) {
    size_t before = stream->size;
    if (drain_pipe(pipe, stream) != NPUNLOCK_STATUS_OK || stream->size == before) {
      break;
    }
  }
}

void npunlock_worker_process_result_release(npunlock_worker_process_result *result) {
  if (result == NULL) {
    return;
  }
  npunlock_buffer_release(&result->response);
  npunlock_buffer_release(&result->stdout_log);
  npunlock_buffer_release(&result->stderr_log);
  memset(result, 0, sizeof(*result));
}

npunlock_status npunlock_worker_process_run(npunlock_view worker_executable_utf8,
                                            const wchar_t *worker_mode, npunlock_view request,
                                            size_t maximum_response_size, uint32_t timeout_ms,
                                            npunlock_worker_process_result *result) {
  SECURITY_ATTRIBUTES security = {sizeof(security), NULL, TRUE};
  STARTUPINFOEXW startup = {0};
  PROCESS_INFORMATION process = {0};
  request_writer writer = {0};
  byte_stream response = {NULL, 0, 0, maximum_response_size, false};
  byte_stream stdout_log = {NULL, 0, 0, NPUNLOCK_WORKER_MAX_LOG_SIZE, false};
  byte_stream stderr_log = {NULL, 0, 0, NPUNLOCK_WORKER_MAX_LOG_SIZE, false};
  HANDLE input_read = NULL;
  HANDLE input_write = NULL;
  HANDLE response_read = NULL;
  HANDLE response_write = NULL;
  HANDLE stdout_read = NULL;
  HANDLE stdout_write = NULL;
  HANDLE stderr_read = NULL;
  HANDLE stderr_write = NULL;
  HANDLE job = NULL;
  HANDLE writer_thread = NULL;
  HANDLE inherited_handles[4];
  SIZE_T attribute_size = 0;
  bool attribute_list_initialized = false;
  wchar_t *worker_path = NULL;
  wchar_t *command_line = NULL;
  size_t command_capacity;
  ULONGLONG deadline;
  bool timed_out = false;
  DWORD exit_code = 0;
  npunlock_status status = NPUNLOCK_STATUS_INTERNAL_ERROR;

  if (result == NULL || worker_mode == NULL || worker_mode[0] == L'\0' ||
      !npunlock_view_is_valid(worker_executable_utf8) || !npunlock_view_is_valid(request) ||
      request.size == 0 || maximum_response_size == 0 || timeout_ms == 0) {
    return NPUNLOCK_STATUS_INVALID_ARGUMENT;
  }
  memset(result, 0, sizeof(*result));
  worker_path = worker_executable_utf8.size == 0 ? default_worker_path()
                                                 : utf8_to_wide(worker_executable_utf8);
  if (worker_path == NULL) {
    status = worker_executable_utf8.size == 0 ? NPUNLOCK_STATUS_INTERNAL_ERROR
                                              : NPUNLOCK_STATUS_INVALID_ARGUMENT;
    goto done;
  }
  if (!CreatePipe(&input_read, &input_write, &security, 0) ||
      !CreatePipe(&response_read, &response_write, &security, 0) ||
      !CreatePipe(&stdout_read, &stdout_write, &security, 0) ||
      !CreatePipe(&stderr_read, &stderr_write, &security, 0) ||
      !SetHandleInformation(input_write, HANDLE_FLAG_INHERIT, 0) ||
      !SetHandleInformation(response_read, HANDLE_FLAG_INHERIT, 0) ||
      !SetHandleInformation(stdout_read, HANDLE_FLAG_INHERIT, 0) ||
      !SetHandleInformation(stderr_read, HANDLE_FLAG_INHERIT, 0)) {
    status = NPUNLOCK_STATUS_INTERNAL_ERROR;
    goto done;
  }
  job = CreateJobObjectW(NULL, NULL);
  if (job == NULL || !configure_job(job)) {
    status = NPUNLOCK_STATUS_INTERNAL_ERROR;
    goto done;
  }
  command_capacity = wcslen(worker_path) + wcslen(worker_mode) + 80;
  command_line = (wchar_t *)malloc(command_capacity * sizeof(*command_line));
  if (command_line == NULL) {
    status = NPUNLOCK_STATUS_OUT_OF_MEMORY;
    goto done;
  }
  swprintf_s(command_line, command_capacity, L"\"%ls\" %ls --response-handle %llu", worker_path,
             worker_mode, (unsigned long long)(uintptr_t)response_write);
  inherited_handles[0] = input_read;
  inherited_handles[1] = response_write;
  inherited_handles[2] = stdout_write;
  inherited_handles[3] = stderr_write;
  InitializeProcThreadAttributeList(NULL, 1, 0, &attribute_size);
  startup.lpAttributeList = (LPPROC_THREAD_ATTRIBUTE_LIST)malloc(attribute_size);
  if (startup.lpAttributeList == NULL) {
    status = NPUNLOCK_STATUS_INTERNAL_ERROR;
    goto done;
  }
  if (!InitializeProcThreadAttributeList(startup.lpAttributeList, 1, 0, &attribute_size)) {
    status = NPUNLOCK_STATUS_INTERNAL_ERROR;
    goto done;
  }
  attribute_list_initialized = true;
  if (!UpdateProcThreadAttribute(startup.lpAttributeList, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                                 inherited_handles, sizeof(inherited_handles), NULL, NULL)) {
    status = NPUNLOCK_STATUS_INTERNAL_ERROR;
    goto done;
  }
  startup.StartupInfo.cb = sizeof(startup);
  startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
  startup.StartupInfo.hStdInput = input_read;
  startup.StartupInfo.hStdOutput = stdout_write;
  startup.StartupInfo.hStdError = stderr_write;
  if (!CreateProcessW(worker_path, command_line, NULL, NULL, TRUE,
                      CREATE_NO_WINDOW | CREATE_SUSPENDED | EXTENDED_STARTUPINFO_PRESENT, NULL,
                      NULL, &startup.StartupInfo, &process)) {
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
    goto finish_process;
  }
  CloseHandle(process.hThread);
  process.hThread = NULL;
  CloseHandle(input_read);
  input_read = NULL;
  CloseHandle(response_write);
  response_write = NULL;
  CloseHandle(stdout_write);
  stdout_write = NULL;
  CloseHandle(stderr_write);
  stderr_write = NULL;

  writer.pipe = input_write;
  writer.data = request.data;
  writer.size = request.size;
  writer_thread = CreateThread(NULL, 0, write_request, &writer, 0, NULL);
  if (writer_thread == NULL) {
    status = NPUNLOCK_STATUS_INTERNAL_ERROR;
    goto terminate;
  }
  input_write = NULL;
  deadline = GetTickCount64() + timeout_ms;
  for (;;) {
    status = drain_pipe(response_read, &response);
    if (status == NPUNLOCK_STATUS_OK) {
      status = drain_pipe(stdout_read, &stdout_log);
    }
    if (status == NPUNLOCK_STATUS_OK) {
      status = drain_pipe(stderr_read, &stderr_log);
    }
    if (status != NPUNLOCK_STATUS_OK) {
      goto terminate;
    }
    if (WaitForSingleObject(process.hProcess, 0) == WAIT_OBJECT_0 && response.closed &&
        stdout_log.closed && stderr_log.closed) {
      break;
    }
    if (GetTickCount64() >= deadline) {
      timed_out = true;
      status = NPUNLOCK_STATUS_TIMEOUT;
      goto terminate;
    }
    Sleep(1);
  }
  if (!GetExitCodeProcess(process.hProcess, &exit_code)) {
    status = NPUNLOCK_STATUS_INTERNAL_ERROR;
    goto terminate;
  }
  result->process_exit_code = exit_code;
  status = exit_code == 0 ? NPUNLOCK_STATUS_OK : NPUNLOCK_STATUS_DRIVER_FAILED;
  goto finish_process;

terminate:
  TerminateJobObject(job, timed_out ? 124 : 125);

finish_process:
  if (process.hProcess != NULL) {
    WaitForSingleObject(process.hProcess, 5000);
    if (GetExitCodeProcess(process.hProcess, &exit_code)) {
      result->process_exit_code = exit_code;
    }
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
  if (response_read != NULL) {
    drain_after_exit(response_read, &response);
  }
  if (stdout_read != NULL) {
    drain_after_exit(stdout_read, &stdout_log);
  }
  if (stderr_read != NULL) {
    drain_after_exit(stderr_read, &stderr_log);
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
  if (stdout_read != NULL) {
    CloseHandle(stdout_read);
  }
  if (stdout_write != NULL) {
    CloseHandle(stdout_write);
  }
  if (stderr_read != NULL) {
    CloseHandle(stderr_read);
  }
  if (stderr_write != NULL) {
    CloseHandle(stderr_write);
  }
  free(worker_path);
  free(command_line);
  if (attribute_list_initialized) {
    DeleteProcThreadAttributeList(startup.lpAttributeList);
  }
  if (startup.lpAttributeList != NULL) {
    free(startup.lpAttributeList);
  }
  if (npunlock_buffer_adopt_malloc(response.data, response.size, &result->response) ==
      NPUNLOCK_STATUS_OK) {
    response.data = NULL;
  }
  if (npunlock_buffer_adopt_malloc(stdout_log.data, stdout_log.size, &result->stdout_log) ==
      NPUNLOCK_STATUS_OK) {
    stdout_log.data = NULL;
  }
  if (npunlock_buffer_adopt_malloc(stderr_log.data, stderr_log.size, &result->stderr_log) ==
      NPUNLOCK_STATUS_OK) {
    stderr_log.data = NULL;
  }
  free(response.data);
  free(stdout_log.data);
  free(stderr_log.data);
  return status;
}
