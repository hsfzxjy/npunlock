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
  /* MVC_DEPEND root containing the required bin/ and lib/ directories. */
  npunlock_view movi_dll_directory_utf8;
  /* Empty selects the installed sibling npunlock_worker executable. */
  npunlock_view worker_executable_utf8;
  npunlock_view target_cpu;
  npunlock_view entry_symbol;
  const npunlock_view *compiler_definitions;
  size_t compiler_definition_count;
  npunlock_view linker_script; /* Empty selects shavecc_default_linker_script(). */
  uint32_t timeout_ms;
} shavecc_options;

typedef struct shavecc_result {
  uint32_t struct_size;
  npunlock_buffer elf;
  /* Verbatim output captured across compiler, assembler, and linker workers. */
  npunlock_buffer stdout_log;
  npunlock_buffer stderr_log;
  npunlock_diagnostic diagnostic;
} shavecc_result;

NPUNLOCK_SHAVECC_API npunlock_status shavecc_compile(const shavecc_options *options,
                                                     npunlock_view c_source,
                                                     shavecc_result *result);
NPUNLOCK_SHAVECC_API void shavecc_result_release(shavecc_result *result);
/* Immutable library-owned bytes valid for the lifetime of the loaded library. */
NPUNLOCK_SHAVECC_API npunlock_view shavecc_default_linker_script(void);
/* Source for the built-in <npunlock/npu3720_kernel.h> virtual include. */
NPUNLOCK_SHAVECC_API npunlock_view shavecc_npu3720_kernel_header(void);

#ifdef __cplusplus
}
#endif

#endif
