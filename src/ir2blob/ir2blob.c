#include "npunlock/ir2blob.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "internal.h"
#include "ir_worker.h"

static int view_has_nul(npunlock_view view) {
  return view.size != 0 && memchr(view.data, 0, view.size) != NULL;
}

static npunlock_status worker_failure(ir2blob_result *result, npunlock_status status,
                                      const npunlock_ir_worker_result *worker) {
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
             "Level Zero worker failed (worker_status=%u, driver_result=0x%08x, "
             "process_exit=0x%08x)",
             worker->worker_status, worker->driver_result, worker->process_exit_code);
  }
  status = npunlock_set_diagnostic(&result->diagnostic, status, "ir2blob.compile",
                                   message != NULL ? message : fallback);
  free(message);
  return status;
}

npunlock_status ir2blob_compile(const ir2blob_options *options, npunlock_view ir_xml,
                                npunlock_view weights, ir2blob_result *result) {
  npunlock_ir_worker_result worker = {0};
  npunlock_status status;
  if (result == NULL) {
    return NPUNLOCK_STATUS_INVALID_ARGUMENT;
  }
  memset(result, 0, sizeof(*result));
  result->struct_size = (uint32_t)sizeof(*result);
  if (options == NULL || options->struct_size < sizeof(*options) ||
      !npunlock_view_is_valid(ir_xml) || ir_xml.size == 0 || !npunlock_view_is_valid(weights) ||
      !npunlock_view_is_valid(options->worker_executable_utf8) ||
      view_has_nul(options->worker_executable_utf8) ||
      !npunlock_view_is_valid(options->build_flags) || view_has_nul(options->build_flags) ||
      options->timeout_ms == 0) {
    return npunlock_set_diagnostic(&result->diagnostic, NPUNLOCK_STATUS_INVALID_ARGUMENT,
                                   "ir2blob.validate", "invalid options, XML, or weights view");
  }
  status = npunlock_run_ir_worker(options->worker_executable_utf8, options->driver_index,
                                  options->device_index, ir_xml, weights, options->build_flags,
                                  options->timeout_ms, &worker);
  result->stdout_log = worker.stdout_log;
  memset(&worker.stdout_log, 0, sizeof(worker.stdout_log));
  result->stderr_log = worker.stderr_log;
  memset(&worker.stderr_log, 0, sizeof(worker.stderr_log));
  if (status != NPUNLOCK_STATUS_OK) {
    status = worker_failure(result, status, &worker);
    goto done;
  }
  result->selected_driver_index = worker.selected_driver_index;
  result->selected_device_index = worker.selected_device_index;
  result->graph_extension_version = worker.graph_extension_version;
  result->compiler_version_major = worker.compiler_version_major;
  result->compiler_version_minor = worker.compiler_version_minor;
  result->max_opset_version = worker.max_opset_version;
  result->driver_version = worker.driver_version;
  result->device_vendor_id = worker.device_vendor_id;
  result->device_id = worker.device_id;
  result->elf_version_major = worker.elf_version_major;
  result->elf_version_minor = worker.elf_version_minor;
  result->elf_version_patch = worker.elf_version_patch;
  result->runtime_version_major = worker.runtime_version_major;
  result->runtime_version_minor = worker.runtime_version_minor;
  result->runtime_version_patch = worker.runtime_version_patch;
  result->graph_blob = worker.graph_blob;
  memset(&worker.graph_blob, 0, sizeof(worker.graph_blob));

done:
  npunlock_ir_worker_result_release(&worker);
  return status;
}

void ir2blob_result_release(ir2blob_result *result) {
  if (result == NULL) {
    return;
  }
  npunlock_buffer_release(&result->graph_blob);
  npunlock_buffer_release(&result->stdout_log);
  npunlock_buffer_release(&result->stderr_log);
  npunlock_diagnostic_release(&result->diagnostic);
  memset(result, 0, sizeof(*result));
}
