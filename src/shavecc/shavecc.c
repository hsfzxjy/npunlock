#include "npunlock/shavecc.h"

#include <stddef.h>
#include <string.h>

#include "internal.h"

npunlock_status shavecc_compile(const shavecc_options *options, npunlock_view c_source,
                                shavecc_result *result) {
  if (result == NULL) {
    return NPUNLOCK_STATUS_INVALID_ARGUMENT;
  }
  memset(result, 0, sizeof(*result));
  result->struct_size = (uint32_t)sizeof(*result);
  if (options == NULL || options->struct_size < sizeof(*options) ||
      !npunlock_view_is_valid(c_source) || c_source.size == 0 ||
      !npunlock_view_is_valid(options->movi_dll_directory_utf8) ||
      options->movi_dll_directory_utf8.size == 0 || !npunlock_view_is_valid(options->target_cpu) ||
      !npunlock_view_is_valid(options->entry_symbol) ||
      !npunlock_view_is_valid(options->compiler_flags) ||
      !npunlock_view_is_valid(options->linker_script) || options->linker_script.size == 0 ||
      options->timeout_ms == 0) {
    return npunlock_set_diagnostic(&result->diagnostic, NPUNLOCK_STATUS_INVALID_ARGUMENT,
                                   "shavecc.validate", "invalid options or C source view");
  }
  return npunlock_set_diagnostic(&result->diagnostic, NPUNLOCK_STATUS_NOT_IMPLEMENTED,
                                 "shavecc.worker", "bounded Movi worker implementation is pending");
}

void shavecc_result_release(shavecc_result *result) {
  if (result == NULL) {
    return;
  }
  npunlock_buffer_release(&result->elf);
  npunlock_diagnostic_release(&result->diagnostic);
  memset(result, 0, sizeof(*result));
}
