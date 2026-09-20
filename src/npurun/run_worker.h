#ifndef NPUNLOCK_NPURUN_RUN_WORKER_H
#define NPUNLOCK_NPURUN_RUN_WORKER_H

#include <stdint.h>

#include "npunlock/buffer.h"
#include "npunlock/error.h"

typedef struct npunlock_run_worker_result {
  npunlock_buffer output;
  npunlock_buffer diagnostic;
  uint32_t worker_status;
  uint32_t driver_result;
  uint32_t process_exit_code;
  uint32_t selected_driver_index;
  uint32_t selected_device_index;
  uint32_t driver_version;
  uint32_t device_vendor_id;
  uint32_t device_id;
  uint64_t element_count;
} npunlock_run_worker_result;

npunlock_status npunlock_run_native_worker(npunlock_view worker_executable_utf8,
                                           npunlock_view graph_blob, uint32_t timeout_ms,
                                           npunlock_run_worker_result *result);

void npunlock_run_worker_result_release(npunlock_run_worker_result *result);

#endif
