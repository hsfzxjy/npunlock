#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "npunlock/error.h"

#include "worker_process.h"

#ifndef NPUNLOCK_POSIX_WORKER_FIXTURE
#error "NPUNLOCK_POSIX_WORKER_FIXTURE must name the worker fixture executable"
#endif

static npunlock_view fixture_path(void) {
  static const char path[] = NPUNLOCK_POSIX_WORKER_FIXTURE;
  return (npunlock_view){(const uint8_t *)path, sizeof(path) - 1};
}

static int buffer_equals(const npunlock_buffer *buffer, const char *expected) {
  size_t size = strlen(expected);
  return buffer->size == size && memcmp(buffer->data, expected, size) == 0;
}

static int test_echo(void) {
  static const uint8_t request[] = "worker request";
  npunlock_worker_process_result result = {0};
  npunlock_status status = npunlock_worker_process_run(
      fixture_path(), "echo", (npunlock_view){request, sizeof(request) - 1}, 1024, 2000, &result);
  int passed = status == NPUNLOCK_STATUS_OK && result.process_exit_code == 0 &&
               result.response.size == sizeof(request) - 1 &&
               memcmp(result.response.data, request, sizeof(request) - 1) == 0 &&
               buffer_equals(&result.stdout_log, "fixture stdout\n") &&
               buffer_equals(&result.stderr_log, "fixture stderr\n");
  if (!passed) {
    fprintf(stderr, "POSIX worker echo failed: %s exit=%u response=%zu stdout=%zu stderr=%zu\n",
            npunlock_status_name(status), result.process_exit_code, result.response.size,
            result.stdout_log.size, result.stderr_log.size);
  }
  npunlock_worker_process_result_release(&result);
  return passed;
}

static int test_timeout(void) {
  static const uint8_t request[] = "timeout request";
  npunlock_worker_process_result result = {0};
  npunlock_status status = npunlock_worker_process_run(
      fixture_path(), "hang", (npunlock_view){request, sizeof(request) - 1}, 1024, 100, &result);
  int passed = status == NPUNLOCK_STATUS_TIMEOUT && result.process_exit_code != 0;
  if (!passed) {
    fprintf(stderr, "POSIX worker timeout failed: %s exit=%u\n", npunlock_status_name(status),
            result.process_exit_code);
  }
  npunlock_worker_process_result_release(&result);
  return passed;
}

int main(void) { return test_echo() && test_timeout() ? 0 : 1; }
