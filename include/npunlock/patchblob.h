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
#define PATCHBLOB_TARGET_V2_ABI_VERSION 2u
#define PATCHBLOB_TARGET_V2_MAX_TENSORS 9u

typedef enum patchblob_contract_flags {
  PATCHBLOB_CONTRACT_STATIC = 1u << 0,
  PATCHBLOB_CONTRACT_DENSE = 1u << 1,
  PATCHBLOB_CONTRACT_FP16 = 1u << 2,
  PATCHBLOB_CONTRACT_CMX = 1u << 3,
  PATCHBLOB_CONTRACT_DISJOINT_OUTPUT = 1u << 4,
  PATCHBLOB_CONTRACT_FP32 = 1u << 5,
  PATCHBLOB_CONTRACT_INPUT_1_SCALAR = 1u << 6
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

typedef enum patchblob_tensor_role_v2 {
  PATCHBLOB_TENSOR_ROLE_INPUT_V2 = 1u,
  PATCHBLOB_TENSOR_ROLE_OUTPUT_V2 = 2u
} patchblob_tensor_role_v2;

typedef enum patchblob_tensor_precision_v2 {
  PATCHBLOB_TENSOR_PRECISION_FP16_V2 = 1u,
  PATCHBLOB_TENSOR_PRECISION_FP32_V2 = 2u
} patchblob_tensor_precision_v2;

typedef enum patchblob_tensor_flags_v2 {
  PATCHBLOB_TENSOR_STATIC_V2 = 1u << 0,
  PATCHBLOB_TENSOR_DENSE_V2 = 1u << 1,
  PATCHBLOB_TENSOR_CMX_V2 = 1u << 2,
  PATCHBLOB_TENSOR_DISJOINT_FROM_INPUTS_V2 = 1u << 3
} patchblob_tensor_flags_v2;

typedef enum patchblob_target_flags_v2 {
  PATCHBLOB_TARGET_UNARY_CONVERSION_V2 = 1u << 0
} patchblob_target_flags_v2;

/*
 * Version 2 describes every observed invocation tensor separately. Inputs
 * occupy consecutive tensor_index values from zero; the sole output follows
 * them with tensor_index zero. The fixed array avoids allocator and pointer
 * ownership in the public ABI.
 */
typedef struct patchblob_tensor_contract_v2 {
  uint32_t struct_size;
  uint32_t role;
  uint32_t tensor_index;
  uint32_t precision;
  uint64_t element_count;
  uint64_t span_bytes;
  uint32_t observed_flags;
} patchblob_tensor_contract_v2;

typedef struct patchblob_target_v2 {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t invocation_index;
  uint32_t range_index;
  uint32_t tensor_count;
  uint32_t target_flags;
  patchblob_tensor_contract_v2 tensors[PATCHBLOB_TARGET_V2_MAX_TENSORS];
} patchblob_target_v2;

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

typedef struct patchblob_discovered_target_v2 {
  uint32_t struct_size;
  uint32_t group_index;
  patchblob_target_v2 target;
} patchblob_discovered_target_v2;

typedef struct patchblob_discovery_result_v2 {
  uint32_t struct_size;
  patchblob_discovered_target_v2 *targets;
  size_t target_count;
  size_t group_count;
  npunlock_diagnostic diagnostic;
} patchblob_discovery_result_v2;

typedef struct patchblob_result {
  uint32_t struct_size;
  npunlock_buffer graph_blob;
  npunlock_buffer report_json;
  npunlock_diagnostic diagnostic;
} patchblob_result;

NPUNLOCK_PATCHBLOB_API npunlock_status
patchblob_patch(const patchblob_options *options, npunlock_view graph_blob, npunlock_view shave_elf,
                const patchblob_target *targets, size_t target_count, patchblob_result *result);
NPUNLOCK_PATCHBLOB_API npunlock_status patchblob_patch_v2(
    const patchblob_options *options, npunlock_view graph_blob, npunlock_view shave_elf,
    const patchblob_target_v2 *targets, size_t target_count, patchblob_result *result);
NPUNLOCK_PATCHBLOB_API void patchblob_result_release(patchblob_result *result);
NPUNLOCK_PATCHBLOB_API npunlock_status
patchblob_discover_targets(npunlock_view graph_blob, patchblob_discovery_result *result);
NPUNLOCK_PATCHBLOB_API void patchblob_discovery_result_release(patchblob_discovery_result *result);
NPUNLOCK_PATCHBLOB_API npunlock_status
patchblob_discover_targets_v2(npunlock_view graph_blob, patchblob_discovery_result_v2 *result);
NPUNLOCK_PATCHBLOB_API void
patchblob_discovery_result_v2_release(patchblob_discovery_result_v2 *result);

#ifdef __cplusplus
}
#endif

#endif
