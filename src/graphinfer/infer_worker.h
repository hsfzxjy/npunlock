#ifndef NPUNLOCK_GRAPHINFER_INFER_WORKER_H
#define NPUNLOCK_GRAPHINFER_INFER_WORKER_H

#include <stddef.h>
#include <stdint.h>

#include "npunlock/buffer.h"
#include "npunlock/error.h"
#include "npunlock/graphinfer.h"

typedef struct npunlock_infer_worker_result {
  graphinfer_output *outputs;
  size_t output_count;
  npunlock_buffer diagnostic;
  uint32_t worker_status;
  uint32_t driver_result;
  uint32_t process_exit_code;
  uint32_t selected_driver_index;
  uint32_t selected_device_index;
  uint32_t driver_version;
  uint32_t device_vendor_id;
  uint32_t device_id;
} npunlock_infer_worker_result;

npunlock_status npunlock_run_infer_worker(npunlock_view worker_executable_utf8,
                                          uint32_t driver_index, uint32_t device_index,
                                          npunlock_view graph_blob, const graphinfer_input *inputs,
                                          size_t input_count, uint32_t timeout_ms,
                                          npunlock_infer_worker_result *result);

void npunlock_infer_worker_result_release(npunlock_infer_worker_result *result);

#endif
