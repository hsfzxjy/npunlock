#ifndef NPUNLOCK_COMMON_WORKER_PROCESS_H
#define NPUNLOCK_COMMON_WORKER_PROCESS_H

#include "npunlock/buffer.h"
#include "npunlock/error.h"
#include <stddef.h>
#include <stdint.h>

typedef struct npunlock_worker_process_result {
  npunlock_buffer response;
  npunlock_buffer stdout_log;
  npunlock_buffer stderr_log;
  uint32_t process_exit_code;
} npunlock_worker_process_result;

npunlock_status npunlock_worker_process_run(npunlock_view worker_executable_utf8,
                                            const char *worker_mode, npunlock_view request,
                                            size_t maximum_response_size, uint32_t timeout_ms,
                                            npunlock_worker_process_result *result);

void npunlock_worker_process_result_release(npunlock_worker_process_result *result);

#endif
