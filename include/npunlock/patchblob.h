#ifndef NPUNLOCK_PATCHBLOB_H
#define NPUNLOCK_PATCHBLOB_H

#include <stddef.h>
#include <stdint.h>

#include "npunlock/buffer.h"
#include "npunlock/error.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PATCHBLOB_UNUSED_INDEX UINT32_MAX

typedef enum patchblob_contract_flags {
  PATCHBLOB_CONTRACT_STATIC = 1u << 0,
  PATCHBLOB_CONTRACT_DENSE = 1u << 1,
  PATCHBLOB_CONTRACT_FP16 = 1u << 2,
  PATCHBLOB_CONTRACT_CMX = 1u << 3,
  PATCHBLOB_CONTRACT_DISJOINT_OUTPUT = 1u << 4
} patchblob_contract_flags;

typedef struct patchblob_target {
  uint32_t struct_size;
  uint32_t invocation_index;
  uint32_t range_index;
  uint32_t expected_input_count;
  uint64_t expected_element_count;
  uint64_t expected_span_bytes;
  uint32_t required_contract_flags;
} patchblob_target;

typedef struct patchblob_options {
  uint32_t struct_size;
  uint32_t image_alignment;
  uint32_t tail_padding;
} patchblob_options;

typedef struct patchblob_discovered_target {
  uint32_t struct_size;
  uint32_t group_index;
  patchblob_target target;
} patchblob_discovered_target;

typedef struct patchblob_discovery_result {
  uint32_t struct_size;
  patchblob_discovered_target *targets;
  size_t target_count;
  size_t group_count;
  npunlock_diagnostic diagnostic;
} patchblob_discovery_result;

typedef struct patchblob_result {
  uint32_t struct_size;
  npunlock_buffer graph_blob;
  npunlock_buffer report_json;
  npunlock_diagnostic diagnostic;
} patchblob_result;

NPUNLOCK_PATCHBLOB_API npunlock_status
patchblob_patch(const patchblob_options *options, npunlock_view graph_blob, npunlock_view shave_elf,
                const patchblob_target *targets, size_t target_count, patchblob_result *result);
NPUNLOCK_PATCHBLOB_API void patchblob_result_release(patchblob_result *result);
NPUNLOCK_PATCHBLOB_API npunlock_status
patchblob_discover_targets(npunlock_view graph_blob, patchblob_discovery_result *result);
NPUNLOCK_PATCHBLOB_API void patchblob_discovery_result_release(patchblob_discovery_result *result);

#ifdef __cplusplus
}
#endif

#endif
