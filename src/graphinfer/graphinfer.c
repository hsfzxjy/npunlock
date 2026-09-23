#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "npunlock/graphinfer.h"

#include "internal.h"
#include "session_internal.h"

#define GRAPHINFER_MAX_INPUTS 64u

static bool view_has_nul(npunlock_view view) {
  return view.size != 0 && memchr(view.data, 0, view.size) != NULL;
}

static bool input_is_valid(const graphinfer_input *input) {
  bool by_index;
  bool by_name;
  if (input->struct_size < sizeof(*input) || !npunlock_view_is_valid(input->argument_name_utf8) ||
      !npunlock_view_is_valid(input->data) || input->data.size == 0 ||
      input->argument_name_utf8.size > UINT32_MAX || view_has_nul(input->argument_name_utf8)) {
    return false;
  }
  by_index = input->argument_index != GRAPHINFER_AUTO_INDEX;
  by_name = input->argument_name_utf8.size != 0;
  return by_index != by_name;
}

void graphinfer_result_release(graphinfer_result *result) {
  size_t index;
  if (result == NULL) {
    return;
  }
  for (index = 0; index < result->output_count; ++index) {
    npunlock_buffer_release(&result->outputs[index].argument_name_utf8);
    npunlock_buffer_release(&result->outputs[index].data);
  }
  free(result->outputs);
  npunlock_diagnostic_release(&result->diagnostic);
  memset(result, 0, sizeof(*result));
}

npunlock_status graphinfer_infer(const graphinfer_options *options, npunlock_view graph_blob,
                                 const graphinfer_input *inputs, size_t input_count,
                                 graphinfer_result *result) {
  graphinfer_session_result session_result = {0};
  npunlock_status status;
  size_t index;
  if (result == NULL) {
    return NPUNLOCK_STATUS_INVALID_ARGUMENT;
  }
  memset(result, 0, sizeof(*result));
  result->struct_size = (uint32_t)sizeof(*result);
  if (options == NULL || options->struct_size < sizeof(*options) || options->timeout_ms == 0 ||
      !npunlock_view_is_valid(graph_blob) || graph_blob.size == 0 || input_count == 0 ||
      input_count > GRAPHINFER_MAX_INPUTS || inputs == NULL) {
    return npunlock_set_diagnostic(&result->diagnostic, NPUNLOCK_STATUS_INVALID_ARGUMENT,
                                   "graphinfer.validate",
                                   "invalid options, graph blob, or input array");
  }
  for (index = 0; index < input_count; ++index) {
    size_t previous;
    if (!input_is_valid(&inputs[index])) {
      return npunlock_set_diagnostic(&result->diagnostic, NPUNLOCK_STATUS_INVALID_ARGUMENT,
                                     "graphinfer.validate", "invalid input tensor descriptor");
    }
    for (previous = 0; previous < index; ++previous) {
      bool same_index = inputs[index].argument_index != GRAPHINFER_AUTO_INDEX &&
                        inputs[index].argument_index == inputs[previous].argument_index;
      bool same_name =
          inputs[index].argument_name_utf8.size != 0 &&
          inputs[index].argument_name_utf8.size == inputs[previous].argument_name_utf8.size &&
          memcmp(inputs[index].argument_name_utf8.data, inputs[previous].argument_name_utf8.data,
                 inputs[index].argument_name_utf8.size) == 0;
      if (same_index || same_name) {
        return npunlock_set_diagnostic(&result->diagnostic, NPUNLOCK_STATUS_INVALID_ARGUMENT,
                                       "graphinfer.validate", "duplicate input selector");
      }
    }
  }
  status = graphinfer_session_create(options, graph_blob, &session_result);
  if (status != NPUNLOCK_STATUS_OK) {
    result->diagnostic = session_result.diagnostic;
    memset(&session_result.diagnostic, 0, sizeof(session_result.diagnostic));
    graphinfer_session_result_release(&session_result);
    return status;
  }
  result->selected_driver_index = session_result.selected_driver_index;
  result->selected_device_index = session_result.selected_device_index;
  result->driver_version = session_result.driver_version;
  result->device_vendor_id = session_result.device_vendor_id;
  result->device_id = session_result.device_id;
  status = npunlock_graphinfer_infer_copied(session_result.session, inputs, input_count, result);
  graphinfer_session_result_release(&session_result);
  return status;
}
