#ifndef NPUNLOCK_SHAVECC_H
#define NPUNLOCK_SHAVECC_H

#include <stdint.h>

#include "npunlock/buffer.h"
#include "npunlock/error.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct shavecc_options {
  uint32_t struct_size;
  npunlock_view movi_dll_directory_utf8;
  npunlock_view target_cpu;
  npunlock_view entry_symbol;
  npunlock_view compiler_flags;
  npunlock_view linker_script;
  uint32_t timeout_ms;
} shavecc_options;

typedef struct shavecc_result {
  uint32_t struct_size;
  npunlock_buffer elf;
  npunlock_diagnostic diagnostic;
} shavecc_result;

NPUNLOCK_SHAVECC_API npunlock_status shavecc_compile(const shavecc_options *options,
                                                     npunlock_view c_source,
                                                     shavecc_result *result);
NPUNLOCK_SHAVECC_API void shavecc_result_release(shavecc_result *result);

#ifdef __cplusplus
}
#endif

#endif
