#include "npunlock/ir2blob.h"

#include <stddef.h>
#include <string.h>

#include "internal.h"

npunlock_status ir2blob_compile(const ir2blob_options *options, npunlock_view ir_xml,
                                npunlock_view weights, ir2blob_result *result) {
  if (result == NULL) {
    return NPUNLOCK_STATUS_INVALID_ARGUMENT;
  }
  memset(result, 0, sizeof(*result));
  result->struct_size = (uint32_t)sizeof(*result);
  if (options == NULL || options->struct_size < sizeof(*options) ||
      !npunlock_view_is_valid(ir_xml) || ir_xml.size == 0 || !npunlock_view_is_valid(weights) ||
      !npunlock_view_is_valid(options->build_flags) || options->timeout_ms == 0) {
    return npunlock_set_diagnostic(&result->diagnostic, NPUNLOCK_STATUS_INVALID_ARGUMENT,
                                   "ir2blob.validate", "invalid options, XML, or weights view");
  }
  return npunlock_set_diagnostic(&result->diagnostic, NPUNLOCK_STATUS_NOT_IMPLEMENTED,
                                 "ir2blob.worker",
                                 "bounded Level Zero worker implementation is pending");
}

void ir2blob_result_release(ir2blob_result *result) {
  if (result == NULL) {
    return;
  }
  npunlock_buffer_release(&result->graph_blob);
  npunlock_diagnostic_release(&result->diagnostic);
  memset(result, 0, sizeof(*result));
}
