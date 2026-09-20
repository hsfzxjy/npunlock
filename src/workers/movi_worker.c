#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "movi_protocol.h"

typedef struct movi_buffer {
  void *data;
  size_t size;
} movi_buffer;

typedef struct movi_buffer_list {
  movi_buffer *items;
  size_t count;
} movi_buffer_list;

typedef int(__cdecl *movi_compile_fn)(int argc, char **argv, void *input, size_t input_size,
                                      void **output, size_t *output_size, void **diagnostic,
                                      size_t *diagnostic_size);
typedef int(__cdecl *movi_assemble_fn)(int argc, char **argv, void *input, uint32_t input_size,
                                       void **output, size_t *output_size, void **diagnostic,
                                       size_t *diagnostic_size);
typedef int(__cdecl *movi_link_fn)(int argc, char **argv, movi_buffer_list *inputs,
                                   movi_buffer *output);
typedef void(__cdecl *movi_cleanup_fn)(void);

typedef struct request_cursor {
  const uint8_t *data;
  size_t size;
  size_t offset;
} request_cursor;

typedef struct worker_request {
  npunlock_movi_stage stage;
  uint8_t *dll_path;
  size_t dll_path_size;
  char **arguments;
  size_t argument_count;
  movi_buffer *inputs;
  size_t input_count;
} worker_request;

typedef struct worker_result {
  npunlock_movi_worker_status status;
  int32_t tool_return;
  uint8_t *output;
  size_t output_size;
  uint8_t *diagnostic;
  size_t diagnostic_size;
} worker_result;

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

static bool checked_add(size_t left, size_t right, size_t *result) {
  if (left > SIZE_MAX - right) {
    return false;
  }
  *result = left + right;
  return true;
}

static bool cursor_read(request_cursor *cursor, size_t size, const uint8_t **value) {
  size_t end;
  if (!checked_add(cursor->offset, size, &end) || end > cursor->size) {
    return false;
  }
  *value = cursor->data + cursor->offset;
  cursor->offset = end;
  return true;
}

static bool cursor_u32(request_cursor *cursor, uint32_t *value) {
  const uint8_t *data;
  if (!cursor_read(cursor, 4, &data)) {
    return false;
  }
  *value = load_u32(data);
  return true;
}

static bool cursor_u64(request_cursor *cursor, uint64_t *value) {
  const uint8_t *data;
  if (!cursor_read(cursor, 8, &data)) {
    return false;
  }
  *value = load_u64(data);
  return true;
}

static bool write_all(HANDLE handle, const uint8_t *data, size_t size) {
  while (size != 0) {
    DWORD chunk = size > MAXDWORD ? MAXDWORD : (DWORD)size;
    DWORD written = 0;
    if (!WriteFile(handle, data, chunk, &written, NULL) || written == 0) {
      return false;
    }
    data += written;
    size -= written;
  }
  return true;
}

static bool read_request(HANDLE handle, uint8_t **data, size_t *size) {
  uint8_t *buffer = NULL;
  size_t used = 0;
  size_t capacity = 0;
  for (;;) {
    uint8_t chunk[16384];
    DWORD received = 0;
    if (!ReadFile(handle, chunk, sizeof(chunk), &received, NULL)) {
      if (GetLastError() == ERROR_BROKEN_PIPE) {
        break;
      }
      free(buffer);
      return false;
    }
    if (received == 0) {
      break;
    }
    if (used > NPUNLOCK_MOVI_MAX_MESSAGE_SIZE - received) {
      free(buffer);
      return false;
    }
    if (used + received > capacity) {
      size_t next = capacity == 0 ? 16384 : capacity;
      uint8_t *replacement;
      while (next < used + received) {
        if (next > NPUNLOCK_MOVI_MAX_MESSAGE_SIZE / 2) {
          next = NPUNLOCK_MOVI_MAX_MESSAGE_SIZE;
          break;
        }
        next *= 2;
      }
      replacement = (uint8_t *)realloc(buffer, next);
      if (replacement == NULL) {
        free(buffer);
        return false;
      }
      buffer = replacement;
      capacity = next;
    }
    memcpy(buffer + used, chunk, received);
    used += received;
  }
  *data = buffer;
  *size = used;
  return true;
}

static void request_release(worker_request *request) {
  size_t index;
  if (request == NULL) {
    return;
  }
  for (index = 0; index < request->argument_count; ++index) {
    free(request->arguments[index]);
  }
  for (index = 0; index < request->input_count; ++index) {
    free(request->inputs[index].data);
  }
  free(request->arguments);
  free(request->inputs);
  free(request->dll_path);
  memset(request, 0, sizeof(*request));
}

static bool parse_request(const uint8_t *data, size_t size, worker_request *request) {
  request_cursor cursor = {data, size, 0};
  const uint8_t *header;
  const uint8_t *bytes;
  uint32_t stage;
  uint32_t argument_count;
  uint32_t input_count;
  uint32_t dll_path_size;
  size_t index;
  memset(request, 0, sizeof(*request));
  if (!cursor_read(&cursor, NPUNLOCK_MOVI_REQUEST_HEADER_SIZE, &header) ||
      load_u32(header) != NPUNLOCK_MOVI_REQUEST_MAGIC ||
      load_u32(header + 4) != NPUNLOCK_MOVI_PROTOCOL_VERSION || load_u32(header + 24) != 0 ||
      load_u32(header + 28) != 0) {
    return false;
  }
  stage = load_u32(header + 8);
  argument_count = load_u32(header + 12);
  input_count = load_u32(header + 16);
  dll_path_size = load_u32(header + 20);
  if (stage < NPUNLOCK_MOVI_STAGE_COMPILE || stage > NPUNLOCK_MOVI_STAGE_LINK ||
      argument_count == 0 || argument_count > 128 || input_count == 0 || input_count > 8 ||
      dll_path_size == 0 || dll_path_size > 32767 || !cursor_read(&cursor, dll_path_size, &bytes) ||
      memchr(bytes, 0, dll_path_size) != NULL) {
    return false;
  }
  request->stage = (npunlock_movi_stage)stage;
  request->dll_path = (uint8_t *)malloc((size_t)dll_path_size + 1);
  request->arguments = (char **)calloc((size_t)argument_count + 1, sizeof(*request->arguments));
  request->inputs = (movi_buffer *)calloc(input_count, sizeof(*request->inputs));
  if (request->dll_path == NULL || request->arguments == NULL || request->inputs == NULL) {
    request_release(request);
    return false;
  }
  memcpy(request->dll_path, bytes, dll_path_size);
  request->dll_path[dll_path_size] = 0;
  request->dll_path_size = dll_path_size;
  request->argument_count = argument_count;
  request->input_count = input_count;
  for (index = 0; index < argument_count; ++index) {
    uint32_t length;
    if (!cursor_u32(&cursor, &length) || length == 0 || length > 65535 ||
        !cursor_read(&cursor, length, &bytes) || memchr(bytes, 0, length) != NULL) {
      request_release(request);
      return false;
    }
    request->arguments[index] = (char *)malloc((size_t)length + 1);
    if (request->arguments[index] == NULL) {
      request_release(request);
      return false;
    }
    memcpy(request->arguments[index], bytes, length);
    request->arguments[index][length] = 0;
  }
  for (index = 0; index < input_count; ++index) {
    uint64_t length;
    uint8_t *copy;
    if (!cursor_u64(&cursor, &length) || length > SIZE_MAX ||
        !cursor_read(&cursor, (size_t)length, &bytes)) {
      request_release(request);
      return false;
    }
    if (length == SIZE_MAX) {
      request_release(request);
      return false;
    }
    copy = (uint8_t *)malloc((size_t)length + 1);
    if (copy == NULL) {
      request_release(request);
      return false;
    }
    if (length != 0) {
      memcpy(copy, bytes, (size_t)length);
    }
    copy[length] = 0;
    request->inputs[index].data = copy;
    request->inputs[index].size = (size_t)length;
  }
  if (cursor.offset != cursor.size ||
      (request->stage == NPUNLOCK_MOVI_STAGE_LINK ? input_count != 2 : input_count != 1)) {
    request_release(request);
    return false;
  }
  return true;
}

static wchar_t *utf8_to_wide(const uint8_t *data, size_t size) {
  int count;
  wchar_t *wide;
  if (size > INT_MAX) {
    return NULL;
  }
  count =
      MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, (const char *)data, (int)size, NULL, 0);
  if (count <= 0) {
    return NULL;
  }
  wide = (wchar_t *)malloc(((size_t)count + 1) * sizeof(*wide));
  if (wide == NULL) {
    return NULL;
  }
  if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, (const char *)data, (int)size, wide,
                          count) != count) {
    free(wide);
    return NULL;
  }
  wide[count] = L'\0';
  return wide;
}

static bool copy_result(const void *source, size_t size, uint8_t **destination) {
  uint8_t *copy;
  if (size > NPUNLOCK_MOVI_MAX_RESULT_SIZE || (size != 0 && source == NULL)) {
    return false;
  }
  if (size == 0) {
    *destination = NULL;
    return true;
  }
  copy = (uint8_t *)malloc(size);
  if (copy == NULL) {
    return false;
  }
  memcpy(copy, source, size);
  *destination = copy;
  return true;
}

static void set_internal_diagnostic(worker_result *result, const char *message) {
  size_t size;
  if (result->diagnostic != NULL || message == NULL) {
    return;
  }
  size = strlen(message);
  if (copy_result(message, size, &result->diagnostic)) {
    result->diagnostic_size = size;
  }
}

static void run_tool(const worker_request *request, worker_result *result) {
  wchar_t *dll_path = NULL;
  wchar_t *dll_directory = NULL;
  wchar_t *separator;
  DLL_DIRECTORY_COOKIE cookie = NULL;
  HMODULE library = NULL;
  FARPROC raw_entry;
  FARPROC raw_cleanup;
  movi_cleanup_fn cleanup = NULL;
  void *output = NULL;
  size_t output_size = 0;
  void *diagnostic = NULL;
  size_t diagnostic_size = 0;
  int return_value = -1;

  memset(result, 0, sizeof(*result));
  result->status = NPUNLOCK_MOVI_WORKER_INTERNAL_ERROR;
  result->tool_return = -1;
  dll_path = utf8_to_wide(request->dll_path, request->dll_path_size);
  if (dll_path == NULL) {
    result->status = NPUNLOCK_MOVI_WORKER_BAD_REQUEST;
    set_internal_diagnostic(result, "Movi DLL path is not valid UTF-8");
    goto done;
  }
  dll_directory = _wcsdup(dll_path);
  if (dll_directory == NULL) {
    result->status = NPUNLOCK_MOVI_WORKER_OUT_OF_MEMORY;
    goto done;
  }
  separator = wcsrchr(dll_directory, L'\\');
  if (separator == NULL) {
    separator = wcsrchr(dll_directory, L'/');
  }
  if (separator == NULL || separator == dll_directory) {
    result->status = NPUNLOCK_MOVI_WORKER_BAD_REQUEST;
    set_internal_diagnostic(result, "Movi DLL path must include an absolute directory");
    goto done;
  }
  *separator = L'\0';
  if (!SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_SYSTEM32 | LOAD_LIBRARY_SEARCH_USER_DIRS)) {
    result->status = NPUNLOCK_MOVI_WORKER_LOAD_FAILED;
    set_internal_diagnostic(result, "failed to constrain the DLL search path");
    goto done;
  }
  cookie = AddDllDirectory(dll_directory);
  if (cookie == NULL) {
    result->status = NPUNLOCK_MOVI_WORKER_LOAD_FAILED;
    set_internal_diagnostic(result, "failed to add the caller-selected Movi DLL directory");
    goto done;
  }
  library = LoadLibraryExW(dll_path, NULL,
                           LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32 |
                               LOAD_LIBRARY_SEARCH_USER_DIRS);
  if (library == NULL) {
    result->status = NPUNLOCK_MOVI_WORKER_LOAD_FAILED;
    set_internal_diagnostic(result, "failed to load the caller-selected Movi DLL");
    goto done;
  }
  raw_cleanup = GetProcAddress(library, "freeResults");
  raw_entry =
      GetProcAddress(library, request->stage == NPUNLOCK_MOVI_STAGE_COMPILE ? "main" : "process");
  if (raw_cleanup == NULL || raw_entry == NULL) {
    result->status = NPUNLOCK_MOVI_WORKER_SYMBOL_MISSING;
    set_internal_diagnostic(result, "required Movi DLL export is missing");
    goto done;
  }
  memcpy(&cleanup, &raw_cleanup, sizeof(cleanup));
  if (request->stage == NPUNLOCK_MOVI_STAGE_ASSEMBLE && request->inputs[0].size > UINT32_MAX) {
    result->status = NPUNLOCK_MOVI_WORKER_BAD_REQUEST;
    set_internal_diagnostic(result, "assembler input exceeds its confirmed uint32 extent");
    goto done;
  }
  if (request->stage == NPUNLOCK_MOVI_STAGE_COMPILE) {
    movi_compile_fn entry = NULL;
    memcpy(&entry, &raw_entry, sizeof(entry));
    return_value =
        entry((int)request->argument_count, request->arguments, request->inputs[0].data,
              request->inputs[0].size, &output, &output_size, &diagnostic, &diagnostic_size);
  } else if (request->stage == NPUNLOCK_MOVI_STAGE_ASSEMBLE) {
    movi_assemble_fn entry = NULL;
    memcpy(&entry, &raw_entry, sizeof(entry));
    return_value = entry((int)request->argument_count, request->arguments, request->inputs[0].data,
                         (uint32_t)request->inputs[0].size, &output, &output_size, &diagnostic,
                         &diagnostic_size);
  } else {
    movi_link_fn entry = NULL;
    struct {
      movi_buffer b;
      movi_buffer _[7];
    } linked = {0};
    struct {
      movi_buffer_list bl;
      movi_buffer_list _[7];
    } inputs = {{request->inputs, request->input_count}, {0}};
    memcpy(&entry, &raw_entry, sizeof(entry));
    return_value = entry((int)request->argument_count, request->arguments, &inputs.bl, &linked.b);
    output = linked.b.data;
    output_size = linked.b.size;
  }
  result->tool_return = return_value;
  if (!copy_result(output, output_size, &result->output) ||
      !copy_result(diagnostic, diagnostic_size, &result->diagnostic)) {
    result->status = NPUNLOCK_MOVI_WORKER_BAD_RESULT;
    set_internal_diagnostic(result, "Movi DLL returned an invalid or oversized result");
    goto cleanup_results;
  }
  result->output_size = output_size;
  result->diagnostic_size = diagnostic_size;
  result->status = NPUNLOCK_MOVI_WORKER_OK;

cleanup_results:
  cleanup();
done:
  /*
   * The retained working invocation leaves each module loaded until worker
   * process teardown. Keep the worker lifecycle aligned with that evidence;
   * process teardown reclaims the module state.
   */
  (void)library;
  if (cookie != NULL) {
    RemoveDllDirectory(cookie);
  }
  free(dll_directory);
  free(dll_path);
}

static bool send_response(HANDLE handle, const worker_result *result) {
  uint8_t header[NPUNLOCK_MOVI_RESPONSE_HEADER_SIZE] = {0};
  store_u32(header, NPUNLOCK_MOVI_RESPONSE_MAGIC);
  store_u32(header + 4, NPUNLOCK_MOVI_PROTOCOL_VERSION);
  store_u32(header + 8, (uint32_t)result->status);
  store_u32(header + 12, (uint32_t)result->tool_return);
  store_u64(header + 16, result->output_size);
  store_u64(header + 24, result->diagnostic_size);
  return write_all(handle, header, sizeof(header)) &&
         write_all(handle, result->output, result->output_size) &&
         write_all(handle, result->diagnostic, result->diagnostic_size);
}

int main(int argc, char **argv) {
  char *handle_end = NULL;
  uint64_t handle_value;
  HANDLE response;
  HANDLE input = GetStdHandle(STD_INPUT_HANDLE);
  uint8_t *request_data = NULL;
  size_t request_size = 0;
  worker_request request;
  worker_result result;
  bool parsed = false;
  int exit_code = 2;

  SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
  if (argc != 3 || strcmp(argv[1], "--response-handle") != 0) {
    return 2;
  }
  handle_value = _strtoui64(argv[2], &handle_end, 10);
  if (handle_value == 0 || handle_end == argv[2] || *handle_end != '\0') {
    return 2;
  }
  response = (HANDLE)(uintptr_t)handle_value;
  memset(&request, 0, sizeof(request));
  memset(&result, 0, sizeof(result));
  result.status = NPUNLOCK_MOVI_WORKER_BAD_REQUEST;
  result.tool_return = -1;
  if (input == NULL || input == INVALID_HANDLE_VALUE) {
    goto done;
  }
  if (!read_request(input, &request_data, &request_size)) {
    set_internal_diagnostic(&result, "failed to read the Movi worker request");
  } else if (!parse_request(request_data, request_size, &request)) {
    set_internal_diagnostic(&result, "malformed Movi worker request");
  } else {
    parsed = true;
    run_tool(&request, &result);
  }
  exit_code = send_response(response, &result) ? 0 : 3;

done:
  if (parsed) {
    request_release(&request);
  }
  free(request_data);
  free(result.output);
  free(result.diagnostic);
  CloseHandle(response);
  return exit_code;
}
