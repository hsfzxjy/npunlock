#ifndef NPUNLOCK_PATCHBLOB_GRAPH_BLOB_H
#define NPUNLOCK_PATCHBLOB_GRAPH_BLOB_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "npunlock/patchblob.h"

typedef struct npunlock_patch_detail {
  uint32_t invocation_index;
  uint32_t range_index;
  uint32_t input_count;
  uint64_t parameter_base;
  uint64_t element_count;
  uint64_t span_bytes;
  uint32_t contract_flags;
  uint32_t target_flags_v2;
  uint32_t tensor_count_v2;
  patchblob_tensor_contract_v2 tensors_v2[PATCHBLOB_TARGET_V2_MAX_TENSORS];
  size_t extent_file_offset;
  size_t addend_file_offset;
  uint32_t old_extent;
  uint32_t new_extent;
  int64_t old_addend;
  int64_t new_addend;
} npunlock_patch_detail;

typedef struct npunlock_patch_summary {
  size_t insertion_file_offset;
  size_t inserted_size;
  size_t image_base;
  size_t image_size;
  size_t old_section_table_offset;
  size_t new_section_table_offset;
  uint32_t tile_count;
  bool section_contents_preserved;
  bool mutation_set_preserved;
  npunlock_patch_detail *details;
  size_t detail_count;
} npunlock_patch_summary;

#define NPUNLOCK_ROUTING_MAX_RANK 15u
#define NPUNLOCK_DPU_MAX_INPUT_ADDRESSES 4u
#define NPUNLOCK_DPU_MAX_OUTPUT_ADDRESSES 4u

/*
 * Private inspection vocabulary. These are validated native observations,
 * not logical graph offsets or a partition/replica classification.
 */
typedef struct npunlock_tensor_routing_observation {
  uint32_t data_symbol;
  int64_t data_addend;
  uint32_t rank;
  int32_t dimensions[NPUNLOCK_ROUTING_MAX_RANK];
  int64_t strides_bits[NPUNLOCK_ROUTING_MAX_RANK];
} npunlock_tensor_routing_observation;

typedef struct npunlock_invocation_routing_observation {
  uint32_t group_index;
  uint32_t tile_candidate_raw;
  uint64_t parameter_base;
  uint64_t wait_mask_raw;
  uint64_t post_mask_raw;
  uint32_t barrier_group_raw;
  uint32_t barrier_mask_raw;
  uint32_t start_after_raw;
  uint32_t clean_after_raw;
  patchblob_target_v2 target;
  npunlock_tensor_routing_observation tensors[PATCHBLOB_TARGET_V2_MAX_TENSORS];
} npunlock_invocation_routing_observation;

typedef struct npunlock_network_output_route_observation {
  uint32_t dma_section_index;
  uint32_t dma_task_index;
  uint32_t cmx_source_symbol_raw;
  int64_t cmx_source_addend_raw;
  uint32_t output_symbol_raw;
  int64_t output_addend_raw;
  uint32_t length_bytes;
} npunlock_network_output_route_observation;

typedef struct npunlock_act_output_slice_candidate_observation {
  uint32_t act_group_index;
  uint32_t act_invocation_index;
  uint32_t act_tensor_record_index;
  uint32_t dma_section_index;
  uint32_t dma_task_index;
  uint32_t route_candidate_count;
  uint32_t cmx_source_symbol_raw;
  int64_t act_data_addend_raw;
  uint64_t act_span_bytes;
  int64_t route_cmx_source_addend_raw;
  uint32_t route_length_bytes;
  uint32_t output_symbol_raw;
  int64_t route_output_addend_raw;
  uint64_t output_slice_addend_candidate;
} npunlock_act_output_slice_candidate_observation;

#define NPUNLOCK_OUTPUT_COVER_ROUTE_AMBIGUITY UINT32_C(1)
#define NPUNLOCK_OUTPUT_COVER_ROUTE_OVERLAP UINT32_C(2)
#define NPUNLOCK_OUTPUT_COVER_SLICE_OVERLAP UINT32_C(4)
#define NPUNLOCK_OUTPUT_COVER_UNION_MISMATCH UINT32_C(8)

typedef struct npunlock_act_output_cover_observation {
  uint32_t output_symbol_raw;
  uint32_t act_group_count;
  uint32_t group_invocation_count;
  uint32_t routed_invocation_count;
  uint32_t unrouted_invocation_count;
  uint32_t route_count;
  uint32_t slice_candidate_count;
  uint32_t unresolved_flags;
  uint64_t route_span_bytes;
  uint64_t slice_span_bytes;
  bool exact_cover;
} npunlock_act_output_cover_observation;

typedef struct npunlock_act_edge_candidate_observation {
  uint32_t producer_group_index;
  uint32_t producer_invocation_index;
  uint32_t producer_tensor_record_index;
  uint32_t consumer_group_index;
  uint32_t consumer_invocation_index;
  uint32_t consumer_tensor_record_index;
  uint32_t consumer_candidate_count;
  uint32_t schedule_hop_count;
  uint32_t data_symbol_raw;
  int64_t data_addend_raw;
  uint64_t span_bytes;
  uint64_t barrier_intersection_raw;
} npunlock_act_edge_candidate_observation;

typedef struct npunlock_dpu_cmx_address_observation {
  uint32_t field_index;
  uint32_t relocation_type;
  uint32_t data_symbol_raw;
  int64_t data_addend_raw;
} npunlock_dpu_cmx_address_observation;

typedef struct npunlock_dpu_invariant_observation {
  uint32_t section_index;
  uint32_t invariant_index;
  uint32_t cluster_raw;
  uint64_t wait_mask_raw;
  uint64_t post_mask_raw;
  uint32_t barrier_group_raw;
  uint32_t barrier_mask_raw;
  uint32_t start_after_raw;
  uint32_t clean_after_raw;
  uint32_t variant_count_raw;
  uint32_t input_dimensions_xyz[3];
  uint32_t input_dtype_raw;
  uint32_t input_dense_raw;
  uint64_t input_element_count;
  uint64_t input_span_bytes;
  uint32_t output_dimensions_xyz[3];
  uint32_t output_dtype_raw;
  uint32_t output_sparse_raw;
  uint64_t output_element_count;
  uint64_t output_span_bytes;
  npunlock_dpu_cmx_address_observation inputs[NPUNLOCK_DPU_MAX_INPUT_ADDRESSES];
  uint32_t input_count;
  npunlock_dpu_cmx_address_observation outputs[NPUNLOCK_DPU_MAX_OUTPUT_ADDRESSES];
  uint32_t output_count;
} npunlock_dpu_invariant_observation;

#define NPUNLOCK_ACT_DPU_DIRECTION_ACT_TO_DPU 1u
#define NPUNLOCK_ACT_DPU_DIRECTION_DPU_TO_ACT 2u

typedef struct npunlock_act_dpu_edge_candidate_observation {
  uint32_t direction;
  uint32_t act_group_index;
  uint32_t act_invocation_index;
  uint32_t act_tensor_record_index;
  uint32_t dpu_invariant_index;
  uint32_t dpu_field_mask_raw;
  uint32_t act_candidate_count;
  uint32_t data_symbol_raw;
  int64_t act_data_addend_raw;
  uint64_t act_span_bytes;
  int64_t dpu_data_addend_raw;
  uint64_t dpu_span_bytes;
  uint64_t barrier_intersection_raw;
} npunlock_act_dpu_edge_candidate_observation;

typedef struct npunlock_graph_routing_observations {
  npunlock_invocation_routing_observation *invocations;
  size_t invocation_count;
  size_t group_count;
  uint32_t tile_count_observed;
  npunlock_network_output_route_observation *network_output_routes;
  size_t network_output_route_count;
  npunlock_act_output_slice_candidate_observation *act_output_slice_candidates;
  size_t act_output_slice_candidate_count;
  npunlock_act_output_cover_observation *act_output_covers;
  size_t act_output_cover_count;
  npunlock_act_edge_candidate_observation *act_edge_candidates;
  size_t act_edge_candidate_count;
  npunlock_dpu_invariant_observation *dpu_invariants;
  size_t dpu_invariant_count;
  npunlock_act_dpu_edge_candidate_observation *act_dpu_edge_candidates;
  size_t act_dpu_edge_candidate_count;
} npunlock_graph_routing_observations;

npunlock_status npunlock_patch_graph_blob(npunlock_view graph_blob, npunlock_view image,
                                          const patchblob_target *targets, size_t target_count,
                                          uint32_t image_alignment, uint32_t tail_padding,
                                          npunlock_buffer *output, npunlock_patch_summary *summary,
                                          npunlock_diagnostic *diagnostic);
npunlock_status npunlock_patch_graph_blob_v2(npunlock_view graph_blob, npunlock_view image,
                                             const patchblob_target_v2 *targets,
                                             size_t target_count, uint32_t image_alignment,
                                             uint32_t tail_padding, npunlock_buffer *output,
                                             npunlock_patch_summary *summary,
                                             npunlock_diagnostic *diagnostic);
void npunlock_patch_summary_release(npunlock_patch_summary *summary);
npunlock_status npunlock_discover_graph_targets(npunlock_view graph_blob,
                                                patchblob_discovered_target **targets,
                                                size_t *target_count, size_t *group_count,
                                                npunlock_diagnostic *diagnostic);
npunlock_status npunlock_discover_graph_targets_v2(npunlock_view graph_blob,
                                                   patchblob_discovered_target_v2 **targets,
                                                   size_t *target_count, size_t *group_count,
                                                   npunlock_diagnostic *diagnostic);
npunlock_status npunlock_inspect_graph_routing(npunlock_view graph_blob,
                                               npunlock_graph_routing_observations *observations,
                                               npunlock_diagnostic *diagnostic);
npunlock_status
npunlock_derive_act_edge_candidates(const npunlock_invocation_routing_observation *invocations,
                                    size_t invocation_count,
                                    npunlock_act_edge_candidate_observation **candidates,
                                    size_t *candidate_count, npunlock_diagnostic *diagnostic);
npunlock_status npunlock_derive_act_output_cover_observations(
    const npunlock_invocation_routing_observation *invocations, size_t invocation_count,
    const npunlock_network_output_route_observation *routes, size_t route_count,
    const npunlock_act_output_slice_candidate_observation *slice_candidates,
    size_t slice_candidate_count, npunlock_act_output_cover_observation **covers,
    size_t *cover_count, npunlock_diagnostic *diagnostic);
void npunlock_graph_routing_observations_release(npunlock_graph_routing_observations *observations);

#endif
