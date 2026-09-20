#include <stdlib.h>
#include <string.h>

#include "infer_worker.h"

npunlock_status npunlock_run_infer_worker(npunlock_view worker_executable_utf8,
                                          uint32_t driver_index, uint32_t device_index,
                                          npunlock_view graph_blob, const graphinfer_input *inputs,
                                          size_t input_count, uint32_t timeout_ms,
                                          npunlock_infer_worker_result *result) {
  (void)worker_executable_utf8;
  (void)driver_index;
  (void)device_index;
  (void)graph_blob;
  (void)inputs;
  (void)input_count;
  (void)timeout_ms;
  if (result == NULL) {
    return NPUNLOCK_STATUS_INVALID_ARGUMENT;
  }
  memset(result, 0, sizeof(*result));
  return NPUNLOCK_STATUS_UNSUPPORTED;
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
