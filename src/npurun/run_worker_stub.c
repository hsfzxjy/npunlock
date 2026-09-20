#include <string.h>

#include "run_worker.h"

npunlock_status npunlock_run_native_worker(npunlock_view worker_executable_utf8,
                                           npunlock_view graph_blob, uint32_t timeout_ms,
                                           npunlock_run_worker_result *result) {
  (void)worker_executable_utf8;
  (void)graph_blob;
  (void)timeout_ms;
  if (result == NULL) {
    return NPUNLOCK_STATUS_INVALID_ARGUMENT;
  }
  memset(result, 0, sizeof(*result));
  return NPUNLOCK_STATUS_UNSUPPORTED;
}

void npunlock_run_worker_result_release(npunlock_run_worker_result *result) {
  if (result == NULL) {
    return;
  }
  npunlock_buffer_release(&result->output);
  npunlock_buffer_release(&result->diagnostic);
  memset(result, 0, sizeof(*result));
}
