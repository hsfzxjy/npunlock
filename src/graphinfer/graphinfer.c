#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "npunlock/graphinfer.h"

#include "infer_worker.h"
#include "internal.h"

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

static npunlock_status worker_failure(graphinfer_result *result, npunlock_status status,
                                      const npunlock_infer_worker_result *worker) {
  char fallback[192];
  char *message = NULL;
  size_t index;
  if (worker->diagnostic.size != 0 && worker->diagnostic.size <= 65535) {
    message = (char *)malloc(worker->diagnostic.size + 1);
    if (message != NULL) {
      memcpy(message, worker->diagnostic.data, worker->diagnostic.size);
      for (index = 0; index < worker->diagnostic.size; ++index) {
        if (message[index] == '\0') {
          message[index] = '?';
        }
      }
      message[worker->diagnostic.size] = '\0';
    }
  }
  if (message == NULL) {
    snprintf(fallback, sizeof(fallback),
             "inference worker failed (worker_status=%u, driver_result=0x%08x, "
             "process_exit=0x%08x)",
             worker->worker_status, worker->driver_result, worker->process_exit_code);
  }
  status = npunlock_set_diagnostic(&result->diagnostic, status, "graphinfer.infer",
                                   message != NULL ? message : fallback);
  free(message);
  return status;
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
  npunlock_buffer_release(&result->stdout_log);
  npunlock_buffer_release(&result->stderr_log);
  npunlock_diagnostic_release(&result->diagnostic);
  memset(result, 0, sizeof(*result));
}

npunlock_status graphinfer_infer(const graphinfer_options *options, npunlock_view graph_blob,
                                 const graphinfer_input *inputs, size_t input_count,
                                 graphinfer_result *result) {
  npunlock_infer_worker_result worker = {0};
  npunlock_status status;
  size_t index;
  if (result == NULL) {
    return NPUNLOCK_STATUS_INVALID_ARGUMENT;
  }
  memset(result, 0, sizeof(*result));
  result->struct_size = (uint32_t)sizeof(*result);
  if (options == NULL || options->struct_size < sizeof(*options) || options->timeout_ms == 0 ||
      !npunlock_view_is_valid(options->worker_executable_utf8) ||
      view_has_nul(options->worker_executable_utf8) || !npunlock_view_is_valid(graph_blob) ||
      graph_blob.size == 0 || input_count == 0 || input_count > GRAPHINFER_MAX_INPUTS ||
      inputs == NULL) {
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
  status = npunlock_run_infer_worker(options->worker_executable_utf8, options->driver_index,
                                     options->device_index, graph_blob, inputs, input_count,
                                     options->timeout_ms, &worker);
  result->stdout_log = worker.stdout_log;
  memset(&worker.stdout_log, 0, sizeof(worker.stdout_log));
  result->stderr_log = worker.stderr_log;
  memset(&worker.stderr_log, 0, sizeof(worker.stderr_log));
  if (status != NPUNLOCK_STATUS_OK) {
    status = worker_failure(result, status, &worker);
    npunlock_infer_worker_result_release(&worker);
    return status;
  }
  result->selected_driver_index = worker.selected_driver_index;
  result->selected_device_index = worker.selected_device_index;
  result->driver_version = worker.driver_version;
  result->device_vendor_id = worker.device_vendor_id;
  result->device_id = worker.device_id;
  result->outputs = worker.outputs;
  result->output_count = worker.output_count;
  worker.outputs = NULL;
  worker.output_count = 0;
  npunlock_infer_worker_result_release(&worker);
  return NPUNLOCK_STATUS_OK;
}
