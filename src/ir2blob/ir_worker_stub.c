#include "ir_worker.h"

#include <string.h>

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
  (void)worker_executable_utf8;
  (void)driver_index;
  (void)device_index;
  (void)ir_xml;
  (void)weights;
  (void)build_flags;
  (void)timeout_ms;
  if (result == NULL) {
    return NPUNLOCK_STATUS_INVALID_ARGUMENT;
  }
  memset(result, 0, sizeof(*result));
  return NPUNLOCK_STATUS_UNSUPPORTED;
}
