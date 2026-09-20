#include "npunlock/patchblob.h"

#include <stddef.h>
#include <string.h>

#include "internal.h"

npunlock_status patchblob_patch(const patchblob_options *options, npunlock_view graph_blob,
                                npunlock_view shave_elf, const patchblob_target *targets,
                                size_t target_count, patchblob_result *result) {
  size_t index;
  if (result == NULL) {
    return NPUNLOCK_STATUS_INVALID_ARGUMENT;
  }
  memset(result, 0, sizeof(*result));
  result->struct_size = (uint32_t)sizeof(*result);
  if (options == NULL || options->struct_size < sizeof(*options) ||
      !npunlock_view_is_valid(graph_blob) || graph_blob.size == 0 ||
      !npunlock_view_is_valid(shave_elf) || shave_elf.size == 0 || targets == NULL ||
      target_count == 0 || options->image_alignment == 0 ||
      (options->image_alignment & (options->image_alignment - 1)) != 0) {
    return npunlock_set_diagnostic(&result->diagnostic, NPUNLOCK_STATUS_INVALID_ARGUMENT,
                                   "patchblob.validate",
                                   "invalid options, input views, or target list");
  }
  for (index = 0; index < target_count; ++index) {
    if (targets[index].struct_size < sizeof(targets[index]) ||
        targets[index].range_index == PATCHBLOB_UNUSED_INDEX ||
        targets[index].expected_input_count == 0 || targets[index].expected_element_count == 0 ||
        targets[index].expected_span_bytes == 0) {
      return npunlock_set_diagnostic(&result->diagnostic, NPUNLOCK_STATUS_INVALID_ARGUMENT,
                                     "patchblob.validate_target",
                                     "each target requires an explicit range and contract");
    }
  }
  return npunlock_set_diagnostic(&result->diagnostic, NPUNLOCK_STATUS_NOT_IMPLEMENTED,
                                 "patchblob.patch",
                                 "bounds-checked graph mutation implementation is pending");
}

void patchblob_result_release(patchblob_result *result) {
  if (result == NULL) {
    return;
  }
  npunlock_buffer_release(&result->graph_blob);
  npunlock_buffer_release(&result->report_json);
  npunlock_diagnostic_release(&result->diagnostic);
  memset(result, 0, sizeof(*result));
}
