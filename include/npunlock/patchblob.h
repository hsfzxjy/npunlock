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
  PATCHBLOB_TENSOR_PRECISION_FP32_V2 = 2u,
  PATCHBLOB_TENSOR_PRECISION_I32_V2 = 3u,
  PATCHBLOB_TENSOR_PRECISION_I8_V2 = 4u,
  PATCHBLOB_TENSOR_PRECISION_I16_V2 = 5u,
  PATCHBLOB_TENSOR_PRECISION_U8_V2 = 6u,
  PATCHBLOB_TENSOR_PRECISION_U16_V2 = 7u,
  PATCHBLOB_TENSOR_PRECISION_U32_V2 = 8u
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

typedef enum patchblob_comparison_flags {
  PATCHBLOB_COMPARE_STRUCTURE_CHANGED = 1u << 0,
  PATCHBLOB_COMPARE_INVOCATION_CONTRACT_CHANGED = 1u << 1,
  PATCHBLOB_COMPARE_OUTPUT_ROUTING_CHANGED = 1u << 2,
  PATCHBLOB_COMPARE_ACT_RELATIONS_CHANGED = 1u << 3,
  PATCHBLOB_COMPARE_DPU_CHANGED = 1u << 4,
  PATCHBLOB_COMPARE_ACT_DPU_RELATIONS_CHANGED = 1u << 5,
  PATCHBLOB_COMPARE_PARTIAL_GROUP_CODE_CHANGE = 1u << 6,
  PATCHBLOB_COMPARE_DMA_CHANGED = 1u << 7,
  PATCHBLOB_COMPARE_ACT_DMA_RELATIONS_CHANGED = 1u << 8,
  PATCHBLOB_COMPARE_ACT_DMA_DPU_RELATIONS_CHANGED = 1u << 9
} patchblob_comparison_flags;

/*
 * Differential evidence for two compiler-produced native graph blobs.
 * contract_stable is true only when all inspected non-code observations are
 * equal. unique_changed_group additionally requires exactly one complete ACT
 * group to have changed code. This result does not establish a symbolic
 * source-node identity and is not a patch selector.
 */
typedef struct patchblob_comparison_result {
  uint32_t struct_size;
  uint32_t mismatch_flags;
  uint32_t contract_stable;
  uint32_t unique_changed_group;
  uint32_t *changed_group_indices;
  size_t changed_group_count;
  npunlock_diagnostic diagnostic;
} patchblob_comparison_result;

typedef enum patchblob_group_completeness_state {
  PATCHBLOB_GROUP_COMPLETENESS_NOT_OBSERVED = 0u,
  PATCHBLOB_GROUP_COMPLETENESS_INCOMPLETE = 1u,
  PATCHBLOB_GROUP_COMPLETENESS_COMPLETE = 2u
} patchblob_group_completeness_state;

typedef enum patchblob_group_completeness_flags {
  PATCHBLOB_GROUP_OUTPUT_SYMBOL_AMBIGUOUS = 1u << 0,
  PATCHBLOB_GROUP_OUTPUT_COVER_UNRESOLVED = 1u << 1,
  PATCHBLOB_GROUP_ROUTE_AMBIGUOUS = 1u << 2,
  PATCHBLOB_GROUP_REPLICA_UNRESOLVED = 1u << 3,
  PATCHBLOB_GROUP_INVOCATION_UNCLASSIFIED = 1u << 4
} patchblob_group_completeness_flags;

/*
 * Direct-output completeness is observed only for groups with a validated
 * route to a graph output. NOT_OBSERVED is deliberately distinct from both
 * COMPLETE and INCOMPLETE; intermediate groups commonly have no such proof.
 */
typedef struct patchblob_group_completeness {
  uint32_t struct_size;
  uint32_t group_index;
  uint32_t state;
  uint32_t output_symbol_raw;
  uint32_t invocation_count;
  uint32_t routed_invocation_count;
  uint32_t replica_invocation_count;
  uint32_t unresolved_invocation_count;
  uint32_t unresolved_flags;
} patchblob_group_completeness;

typedef struct patchblob_group_completeness_result {
  uint32_t struct_size;
  patchblob_group_completeness *groups;
  size_t group_count;
  npunlock_diagnostic diagnostic;
} patchblob_group_completeness_result;

typedef enum patchblob_group_coverage_state {
  PATCHBLOB_GROUP_COVERAGE_NOT_OBSERVED = 0u,
  PATCHBLOB_GROUP_COVERAGE_INCOMPLETE = 1u,
  PATCHBLOB_GROUP_COVERAGE_COMPLETE = 2u
} patchblob_group_coverage_state;

/*
 * Independent native coverage evidence for every ACT group. Each route kind
 * has its own state and unresolved flags; callers must not combine a COMPLETE
 * state with source-label evidence implicitly. This record is diagnostic and
 * is not a patch selector.
 */
typedef struct patchblob_group_coverage {
  uint32_t struct_size;
  uint32_t group_index;
  uint32_t invocation_count;
  uint32_t direct_output_state;
  uint32_t act_consumer_state;
  uint32_t dpu_consumer_state;
  uint32_t direct_output_unresolved_flags;
  uint32_t act_consumer_unresolved_flags;
  uint32_t dpu_consumer_unresolved_flags;
  uint32_t direct_routed_invocation_count;
  uint32_t direct_replica_invocation_count;
  uint32_t act_matched_invocation_count;
  uint32_t act_consumer_group_count;
  uint32_t act_consumer_input_count;
  uint32_t dpu_matched_invocation_count;
  uint32_t dpu_input_count;
} patchblob_group_coverage;

typedef struct patchblob_group_coverage_result {
  uint32_t struct_size;
  patchblob_group_coverage *groups;
  size_t group_count;
  npunlock_diagnostic diagnostic;
} patchblob_group_coverage_result;

typedef enum patchblob_dma_dpu_group_coverage_flags {
  PATCHBLOB_DMA_DPU_FANIN_UNRESOLVED = 1u << 0,
  PATCHBLOB_DMA_DPU_TILE_SET_UNRESOLVED = 1u << 1,
  PATCHBLOB_DMA_DPU_INPUT_AMBIGUOUS = 1u << 2,
  PATCHBLOB_DMA_DPU_INPUT_REUSED = 1u << 3
} patchblob_dma_dpu_group_coverage_flags;

/*
 * DMA-mediated DPU-consumer evidence is separate from direct ACT-to-DPU
 * coverage. It requires an exact ACT-to-DDR-to-multicast-CMX fan-in and a
 * unique translated DPU input for every observed tile.
 */
typedef struct patchblob_dma_dpu_group_coverage {
  uint32_t struct_size;
  uint32_t group_index;
  uint32_t invocation_count;
  uint32_t state;
  uint32_t unresolved_flags;
  uint32_t tile_count_expected;
  uint32_t translated_tile_count;
  uint32_t matched_dpu_input_count;
} patchblob_dma_dpu_group_coverage;

typedef struct patchblob_dma_dpu_group_coverage_result {
  uint32_t struct_size;
  patchblob_dma_dpu_group_coverage *groups;
  size_t group_count;
  npunlock_diagnostic diagnostic;
} patchblob_dma_dpu_group_coverage_result;

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
NPUNLOCK_PATCHBLOB_API npunlock_status
patchblob_compare_graphs(npunlock_view baseline_graph_blob, npunlock_view candidate_graph_blob,
                         patchblob_comparison_result *result);
NPUNLOCK_PATCHBLOB_API void
patchblob_comparison_result_release(patchblob_comparison_result *result);
NPUNLOCK_PATCHBLOB_API npunlock_status patchblob_inspect_group_completeness(
    npunlock_view graph_blob, patchblob_group_completeness_result *result);
NPUNLOCK_PATCHBLOB_API void
patchblob_group_completeness_result_release(patchblob_group_completeness_result *result);
NPUNLOCK_PATCHBLOB_API npunlock_status
patchblob_inspect_group_coverage(npunlock_view graph_blob, patchblob_group_coverage_result *result);
NPUNLOCK_PATCHBLOB_API void
patchblob_group_coverage_result_release(patchblob_group_coverage_result *result);
NPUNLOCK_PATCHBLOB_API npunlock_status patchblob_inspect_dma_dpu_group_coverage(
    npunlock_view graph_blob, patchblob_dma_dpu_group_coverage_result *result);
NPUNLOCK_PATCHBLOB_API void
patchblob_dma_dpu_group_coverage_result_release(patchblob_dma_dpu_group_coverage_result *result);

#ifdef __cplusplus
}
#endif

#endif
