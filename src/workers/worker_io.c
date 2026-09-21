#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "worker_io.h"

bool npunlock_worker_checked_add(size_t left, size_t right, size_t *result) {
  if (left > SIZE_MAX - right) {
    return false;
  }
  *result = left + right;
  return true;
}

bool npunlock_worker_checked_mul(size_t left, size_t right, size_t *result) {
  if (left != 0 && right > SIZE_MAX / left) {
    return false;
  }
  *result = left * right;
  return true;
}

bool npunlock_worker_write_all(HANDLE handle, const uint8_t *data, size_t size) {
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

bool npunlock_worker_read_all(HANDLE handle, size_t maximum, uint8_t **data, size_t *size) {
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
    if (used > maximum - received) {
      free(buffer);
      return false;
    }
    if (used + received > capacity) {
      size_t next = capacity == 0 ? 16384 : capacity;
      uint8_t *replacement;
      while (next < used + received) {
        if (next > maximum / 2) {
          next = maximum;
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

bool npunlock_worker_response_handle(int argc, char **argv, HANDLE *response) {
  char *handle_end = NULL;
  uint64_t handle_value;
  if (response == NULL || argc != 3 || strcmp(argv[1], "--response-handle") != 0) {
    return false;
  }
  handle_value = _strtoui64(argv[2], &handle_end, 10);
  if (handle_value == 0 || handle_end == argv[2] || *handle_end != '\0') {
    return false;
  }
  *response = (HANDLE)(uintptr_t)handle_value;
  return true;
}
