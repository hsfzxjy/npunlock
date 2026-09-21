#ifndef NPUNLOCK_IR2BLOB_IR_WORKER_H
#define NPUNLOCK_IR2BLOB_IR_WORKER_H

#include <stdint.h>

#include "npunlock/buffer.h"
#include "npunlock/error.h"

typedef struct npunlock_ir_worker_result {
  npunlock_buffer graph_blob;
  npunlock_buffer diagnostic;
  npunlock_buffer stdout_log;
  npunlock_buffer stderr_log;
  uint32_t worker_status;
  uint32_t driver_result;
  uint32_t process_exit_code;
  uint32_t selected_driver_index;
  uint32_t selected_device_index;
  uint32_t graph_extension_version;
  uint16_t compiler_version_major;
  uint16_t compiler_version_minor;
  uint32_t max_opset_version;
  uint32_t driver_version;
  uint32_t device_vendor_id;
  uint32_t device_id;
  uint32_t elf_version_major;
  uint32_t elf_version_minor;
  uint32_t elf_version_patch;
  uint32_t runtime_version_major;
  uint32_t runtime_version_minor;
  uint32_t runtime_version_patch;
} npunlock_ir_worker_result;

npunlock_status npunlock_run_ir_worker(npunlock_view worker_executable_utf8, uint32_t driver_index,
                                       uint32_t device_index, npunlock_view ir_xml,
                                       npunlock_view weights, npunlock_view build_flags,
                                       uint32_t timeout_ms, npunlock_ir_worker_result *result);

void npunlock_ir_worker_result_release(npunlock_ir_worker_result *result);

#endif
