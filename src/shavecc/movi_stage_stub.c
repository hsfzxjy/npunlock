#include "movi_stage.h"

#include <string.h>

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
  (void)worker_executable_utf8;
  (void)dll_path_utf8;
  (void)stage;
  (void)arguments;
  (void)argument_count;
  (void)inputs;
  (void)input_count;
  (void)timeout_ms;
  if (result == NULL) {
    return NPUNLOCK_STATUS_INVALID_ARGUMENT;
  }
  memset(result, 0, sizeof(*result));
  return NPUNLOCK_STATUS_UNSUPPORTED;
}
