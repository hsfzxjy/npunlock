#ifndef NPUNLOCK_SHAVECC_MOVI_STAGE_H
#define NPUNLOCK_SHAVECC_MOVI_STAGE_H

#include <stddef.h>
#include <stdint.h>

#include "npunlock/buffer.h"
#include "npunlock/error.h"
#include "workers/movi_protocol.h"

typedef struct npunlock_movi_result {
  npunlock_buffer output;
  npunlock_buffer diagnostic;
  int32_t tool_return;
  uint32_t worker_status;
  uint32_t process_exit_code;
} npunlock_movi_result;

npunlock_status npunlock_run_movi_stage(npunlock_view worker_executable_utf8,
                                        npunlock_view dll_path_utf8, npunlock_movi_stage stage,
                                        const npunlock_view *arguments, size_t argument_count,
                                        const npunlock_view *inputs, size_t input_count,
                                        uint32_t timeout_ms, npunlock_movi_result *result);

void npunlock_movi_result_release(npunlock_movi_result *result);

#endif
