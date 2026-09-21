#ifndef NPUNLOCK_IR2BLOB_H
#define NPUNLOCK_IR2BLOB_H

#include <stdint.h>

#include "npunlock/buffer.h"
#include "npunlock/error.h"

#ifdef __cplusplus
extern "C" {
#endif

#define IR2BLOB_AUTO_INDEX UINT32_MAX

typedef struct ir2blob_options {
  uint32_t struct_size;
  uint32_t driver_index;
  uint32_t device_index;
  uint32_t timeout_ms;
  /* Empty selects the installed sibling npunlock_worker executable. */
  npunlock_view worker_executable_utf8;
  npunlock_view build_flags;
} ir2blob_options;

typedef struct ir2blob_result {
  uint32_t struct_size;
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
  npunlock_buffer graph_blob;
  /* Verbatim output captured from the bounded graph-compiler worker. */
  npunlock_buffer stdout_log;
  npunlock_buffer stderr_log;
  npunlock_diagnostic diagnostic;
} ir2blob_result;

NPUNLOCK_IR2BLOB_API npunlock_status ir2blob_compile(const ir2blob_options *options,
                                                     npunlock_view ir_xml, npunlock_view weights,
                                                     ir2blob_result *result);
NPUNLOCK_IR2BLOB_API void ir2blob_result_release(ir2blob_result *result);

#ifdef __cplusplus
}
#endif

#endif
