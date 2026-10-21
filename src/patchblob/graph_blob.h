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
#define NPUNLOCK_DPU_MAX_INPUT_RELOCATIONS 8u
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
  uint64_t code_offset_raw;
  uint32_t code_extent;
  uint8_t code_sha256[32];
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

#define NPUNLOCK_DMA_ENDPOINT_UNKNOWN 0u
#define NPUNLOCK_DMA_ENDPOINT_CMX 1u
#define NPUNLOCK_DMA_ENDPOINT_DDR_BUFFER 2u
#define NPUNLOCK_DMA_ENDPOINT_CONSTANT 3u
#define NPUNLOCK_DMA_ENDPOINT_NETWORK_INPUT 4u
#define NPUNLOCK_DMA_ENDPOINT_NETWORK_OUTPUT 5u
#define NPUNLOCK_DMA_ENDPOINT_RUNTIME 6u

typedef struct npunlock_dma_endpoint_observation {
  uint32_t memory_kind;
  uint32_t relocation_section_index;
  uint32_t symbol_table_index;
  uint32_t symbol_raw;
  uint32_t relocation_type;
  int64_t addend_raw;
} npunlock_dma_endpoint_observation;

typedef struct npunlock_dma_task_observation {
  uint32_t section_index;
  uint32_t task_index;
  uint32_t engine_index;
  uint32_t transfer_type_raw;
  uint32_t length_bytes;
  uint32_t num_planes_raw;
  uint32_t task_id_raw;
  int32_t source_plane_stride_raw;
  int32_t destination_plane_stride_raw;
  uint32_t source_width_raw;
  int32_t source_stride_raw;
  uint32_t destination_width_raw;
  int32_t destination_stride_raw;
  uint64_t wait_mask_raw;
  uint64_t post_mask_raw;
  uint32_t start_after_raw;
  uint32_t clean_after_raw;
  uint64_t source_value_raw;
  uint64_t destination_value_raw;
  npunlock_dma_endpoint_observation source;
  npunlock_dma_endpoint_observation destination;
} npunlock_dma_task_observation;

typedef struct npunlock_act_dma_source_candidate_observation {
  uint32_t act_group_index;
  uint32_t act_invocation_index;
  uint32_t act_tensor_record_index;
  uint32_t dma_section_index;
  uint32_t dma_task_index;
  uint32_t dma_engine_index;
  uint32_t dma_candidate_count;
  uint32_t ddr_symbol_table_index;
  uint32_t ddr_symbol_raw;
  int64_t act_data_addend_raw;
  int64_t dma_source_addend_raw;
  int64_t ddr_destination_addend_raw;
  uint64_t span_bytes;
  uint64_t barrier_intersection_raw;
  uint64_t dma_post_mask_raw;
} npunlock_act_dma_source_candidate_observation;

#define NPUNLOCK_ACT_DMA_FANIN_SOURCE_UNCLASSIFIED (1u << 0)
#define NPUNLOCK_ACT_DMA_FANIN_SOURCE_AMBIGUOUS (1u << 1)
#define NPUNLOCK_ACT_DMA_FANIN_SOURCE_TASK_REUSED (1u << 2)
#define NPUNLOCK_ACT_DMA_FANIN_DDR_COVER_UNRESOLVED (1u << 3)
#define NPUNLOCK_ACT_DMA_FANIN_TASK_AMBIGUOUS (1u << 4)

typedef struct npunlock_act_dma_fanin_observation {
  uint32_t act_group_index;
  uint32_t invocation_count;
  uint32_t matched_invocation_count;
  uint32_t source_dma_task_count;
  uint32_t fanin_candidate_count;
  uint32_t fanin_dma_section_index;
  uint32_t fanin_dma_task_index;
  uint32_t fanin_dma_engine_index;
  uint32_t ddr_symbol_table_index;
  uint32_t ddr_symbol_raw;
  uint32_t cmx_destination_symbol_table_index;
  uint32_t cmx_destination_symbol_raw;
  int64_t ddr_source_addend_raw;
  uint64_t span_bytes;
  int64_t cmx_destination_addend_raw;
  uint64_t cmx_destination_value_raw;
  uint64_t fanin_wait_mask_raw;
  uint64_t fanin_post_mask_raw;
  uint32_t unresolved_flags;
  bool exact_cover;
} npunlock_act_dma_fanin_observation;

typedef struct npunlock_act_dma_act_input_candidate_observation {
  uint32_t producer_group_index;
  uint32_t producer_invocation_index;
  uint32_t spill_dma_section_index;
  uint32_t spill_dma_task_index;
  uint32_t consumer_group_index;
  uint32_t consumer_invocation_index;
  uint32_t consumer_tensor_record_index;
  uint32_t reload_dma_section_index;
  uint32_t reload_dma_task_index;
  uint32_t reload_dma_engine_index;
  uint32_t consumer_candidate_count;
  uint32_t ddr_symbol_table_index;
  uint32_t ddr_symbol_raw;
  int64_t ddr_source_addend_raw;
  uint32_t cmx_destination_symbol_raw;
  int64_t cmx_destination_addend_raw;
  uint64_t span_bytes;
  uint64_t spill_barrier_intersection_raw;
  uint64_t consumer_barrier_intersection_raw;
} npunlock_act_dma_act_input_candidate_observation;

#define NPUNLOCK_ACT_DMA_ACT_SOURCE_UNCLASSIFIED UINT32_C(1)
#define NPUNLOCK_ACT_DMA_ACT_SOURCE_AMBIGUOUS UINT32_C(2)
#define NPUNLOCK_ACT_DMA_ACT_REPLICA_UNRESOLVED UINT32_C(4)
#define NPUNLOCK_ACT_DMA_ACT_DDR_COVER_UNRESOLVED UINT32_C(8)
#define NPUNLOCK_ACT_DMA_ACT_CONSUMER_AMBIGUOUS UINT32_C(16)
#define NPUNLOCK_ACT_DMA_ACT_RELOAD_TASK_REUSED UINT32_C(32)

typedef struct npunlock_act_dma_act_input_cover_observation {
  uint32_t producer_group_index;
  uint32_t producer_invocation_count;
  uint32_t routed_producer_invocation_count;
  uint32_t replica_producer_invocation_count;
  uint32_t consumer_group_count;
  uint32_t consumer_input_count;
  uint32_t unresolved_flags;
  uint64_t ddr_span_bytes;
  bool exact_cover;
} npunlock_act_dma_act_input_cover_observation;

typedef struct npunlock_act_dma_dpu_input_candidate_observation {
  uint32_t act_group_index;
  uint32_t tile_candidate_raw;
  uint32_t act_invocation_index;
  uint32_t dpu_section_index;
  uint32_t dpu_invariant_index;
  uint32_t dpu_field_mask_raw;
  uint32_t dpu_candidate_count;
  uint32_t cmx_symbol_raw;
  int64_t fanin_destination_addend_raw;
  int64_t tile_translation_candidate;
  int64_t translated_destination_addend_raw;
  int64_t dpu_input_addend_raw;
  uint64_t span_bytes;
  uint64_t barrier_intersection_raw;
} npunlock_act_dma_dpu_input_candidate_observation;

#define NPUNLOCK_ACT_DMA_DPU_FANIN_UNRESOLVED UINT32_C(1)
#define NPUNLOCK_ACT_DMA_DPU_TILE_SET_UNRESOLVED UINT32_C(2)
#define NPUNLOCK_ACT_DMA_DPU_INPUT_AMBIGUOUS UINT32_C(4)
#define NPUNLOCK_ACT_DMA_DPU_INPUT_REUSED UINT32_C(8)

typedef struct npunlock_act_dma_dpu_input_cover_observation {
  uint32_t act_group_index;
  uint32_t tile_count_expected;
  uint32_t invocation_count;
  uint32_t translated_tile_count;
  uint32_t matched_dpu_input_count;
  uint32_t unresolved_flags;
  bool exact_cover;
} npunlock_act_dma_dpu_input_cover_observation;

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

typedef struct npunlock_act_replica_group_observation {
  uint32_t act_group_index;
  uint32_t routed_invocation_count;
  uint32_t unrouted_invocation_count;
  uint32_t translation_solution_count;
  int64_t cmx_translation_candidate;
} npunlock_act_replica_group_observation;

typedef struct npunlock_act_replica_candidate_observation {
  uint32_t act_group_index;
  uint32_t unrouted_invocation_index;
  uint32_t routed_invocation_index;
  uint32_t unrouted_tile_candidate_raw;
  uint32_t routed_tile_candidate_raw;
  uint32_t output_symbol_raw;
  uint64_t output_slice_addend_candidate;
  uint64_t output_slice_span_bytes;
  int64_t cmx_translation_candidate;
} npunlock_act_replica_candidate_observation;

#define NPUNLOCK_GROUP_COMPLETE_OUTPUT_SYMBOL_AMBIGUOUS PATCHBLOB_GROUP_OUTPUT_SYMBOL_AMBIGUOUS
#define NPUNLOCK_GROUP_COMPLETE_OUTPUT_COVER_UNRESOLVED PATCHBLOB_GROUP_OUTPUT_COVER_UNRESOLVED
#define NPUNLOCK_GROUP_COMPLETE_ROUTE_AMBIGUOUS PATCHBLOB_GROUP_ROUTE_AMBIGUOUS
#define NPUNLOCK_GROUP_COMPLETE_REPLICA_UNRESOLVED PATCHBLOB_GROUP_REPLICA_UNRESOLVED
#define NPUNLOCK_GROUP_COMPLETE_INVOCATION_UNCLASSIFIED PATCHBLOB_GROUP_INVOCATION_UNCLASSIFIED

typedef struct npunlock_act_group_completeness_observation {
  uint32_t act_group_index;
  uint32_t output_symbol_raw;
  uint32_t invocation_count;
  uint32_t routed_invocation_count;
  uint32_t replica_invocation_count;
  uint32_t unresolved_invocation_count;
  uint32_t unresolved_flags;
  bool complete;
} npunlock_act_group_completeness_observation;

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

#define NPUNLOCK_ACT_CONSUMER_COVER_EDGE_AMBIGUOUS UINT32_C(1)
#define NPUNLOCK_ACT_CONSUMER_COVER_PRODUCER_UNCLASSIFIED UINT32_C(2)
#define NPUNLOCK_ACT_CONSUMER_COVER_INPUT_DUPLICATED UINT32_C(4)

typedef struct npunlock_act_consumer_cover_observation {
  uint32_t producer_group_index;
  uint32_t producer_invocation_count;
  uint32_t matched_producer_invocation_count;
  uint32_t unresolved_producer_invocation_count;
  uint32_t consumer_group_count;
  uint32_t consumer_input_count;
  uint32_t unresolved_flags;
  bool exact_cover;
} npunlock_act_consumer_cover_observation;

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
  npunlock_dpu_cmx_address_observation inputs[NPUNLOCK_DPU_MAX_INPUT_RELOCATIONS];
  uint32_t input_count;
  uint32_t input_ambiguous_fields_raw;
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
  uint32_t dpu_section_index;
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

#define NPUNLOCK_ACT_DPU_COVER_CONSUMER_AMBIGUOUS UINT32_C(1)
#define NPUNLOCK_ACT_DPU_COVER_INVOCATION_UNCLASSIFIED UINT32_C(2)
#define NPUNLOCK_ACT_DPU_COVER_SLICE_OVERLAP UINT32_C(4)
#define NPUNLOCK_ACT_DPU_COVER_UNION_MISMATCH UINT32_C(8)

typedef struct npunlock_act_dpu_output_cover_observation {
  uint32_t act_group_index;
  uint32_t invocation_count;
  uint32_t matched_invocation_count;
  uint32_t unresolved_invocation_count;
  uint32_t dpu_input_count;
  uint32_t exact_cover_input_count;
  uint32_t unresolved_flags;
  bool exact_cover;
} npunlock_act_dpu_output_cover_observation;

typedef struct npunlock_graph_routing_observations {
  npunlock_invocation_routing_observation *invocations;
  size_t invocation_count;
  size_t group_count;
  uint32_t tile_count_observed;
  npunlock_dma_task_observation *dma_tasks;
  size_t dma_task_count;
  npunlock_act_dma_source_candidate_observation *act_dma_source_candidates;
  size_t act_dma_source_candidate_count;
  npunlock_act_dma_fanin_observation *act_dma_fanin_observations;
  size_t act_dma_fanin_observation_count;
  npunlock_act_dma_act_input_candidate_observation *act_dma_act_input_candidates;
  size_t act_dma_act_input_candidate_count;
  npunlock_act_dma_act_input_cover_observation *act_dma_act_input_covers;
  size_t act_dma_act_input_cover_count;
  npunlock_act_dma_dpu_input_candidate_observation *act_dma_dpu_input_candidates;
  size_t act_dma_dpu_input_candidate_count;
  npunlock_act_dma_dpu_input_cover_observation *act_dma_dpu_input_covers;
  size_t act_dma_dpu_input_cover_count;
  npunlock_network_output_route_observation *network_output_routes;
  size_t network_output_route_count;
  npunlock_act_output_slice_candidate_observation *act_output_slice_candidates;
  size_t act_output_slice_candidate_count;
  npunlock_act_output_cover_observation *act_output_covers;
  size_t act_output_cover_count;
  npunlock_act_replica_group_observation *act_replica_groups;
  size_t act_replica_group_count;
  npunlock_act_replica_candidate_observation *act_replica_candidates;
  size_t act_replica_candidate_count;
  npunlock_act_group_completeness_observation *act_group_completeness;
  size_t act_group_completeness_count;
  npunlock_act_edge_candidate_observation *act_edge_candidates;
  size_t act_edge_candidate_count;
  npunlock_act_consumer_cover_observation *act_consumer_covers;
  size_t act_consumer_cover_count;
  npunlock_dpu_invariant_observation *dpu_invariants;
  size_t dpu_invariant_count;
  npunlock_act_dpu_edge_candidate_observation *act_dpu_edge_candidates;
  size_t act_dpu_edge_candidate_count;
  npunlock_act_dpu_output_cover_observation *act_dpu_output_covers;
  size_t act_dpu_output_cover_count;
} npunlock_graph_routing_observations;

#define NPUNLOCK_COMPARE_STRUCTURE_CHANGED PATCHBLOB_COMPARE_STRUCTURE_CHANGED
#define NPUNLOCK_COMPARE_INVOCATION_CONTRACT_CHANGED PATCHBLOB_COMPARE_INVOCATION_CONTRACT_CHANGED
#define NPUNLOCK_COMPARE_OUTPUT_ROUTING_CHANGED PATCHBLOB_COMPARE_OUTPUT_ROUTING_CHANGED
#define NPUNLOCK_COMPARE_ACT_RELATIONS_CHANGED PATCHBLOB_COMPARE_ACT_RELATIONS_CHANGED
#define NPUNLOCK_COMPARE_DPU_CHANGED PATCHBLOB_COMPARE_DPU_CHANGED
#define NPUNLOCK_COMPARE_ACT_DPU_RELATIONS_CHANGED PATCHBLOB_COMPARE_ACT_DPU_RELATIONS_CHANGED
#define NPUNLOCK_COMPARE_PARTIAL_GROUP_CODE_CHANGE PATCHBLOB_COMPARE_PARTIAL_GROUP_CODE_CHANGE
#define NPUNLOCK_COMPARE_DMA_CHANGED PATCHBLOB_COMPARE_DMA_CHANGED
#define NPUNLOCK_COMPARE_ACT_DMA_RELATIONS_CHANGED PATCHBLOB_COMPARE_ACT_DMA_RELATIONS_CHANGED
#define NPUNLOCK_COMPARE_ACT_DMA_DPU_RELATIONS_CHANGED                                             \
  PATCHBLOB_COMPARE_ACT_DMA_DPU_RELATIONS_CHANGED
#define NPUNLOCK_COMPARE_ACT_DMA_ACT_RELATIONS_CHANGED                                             \
  PATCHBLOB_COMPARE_ACT_DMA_ACT_RELATIONS_CHANGED

typedef struct npunlock_graph_routing_comparison {
  uint32_t mismatch_flags;
  uint32_t *changed_group_indices;
  size_t changed_group_count;
  bool contract_stable;
  bool unique_changed_group;
} npunlock_graph_routing_comparison;

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
npunlock_status npunlock_derive_act_dma_source_candidates(
    const npunlock_invocation_routing_observation *invocations, size_t invocation_count,
    const npunlock_dma_task_observation *dma_tasks, size_t dma_task_count,
    npunlock_act_dma_source_candidate_observation **candidates, size_t *candidate_count,
    npunlock_diagnostic *diagnostic);
npunlock_status npunlock_derive_act_dma_fanin_observations(
    const npunlock_invocation_routing_observation *invocations, size_t invocation_count,
    size_t group_count, const npunlock_dma_task_observation *dma_tasks, size_t dma_task_count,
    const npunlock_act_dma_source_candidate_observation *source_candidates,
    size_t source_candidate_count, npunlock_act_dma_fanin_observation **observations,
    size_t *observation_count, npunlock_diagnostic *diagnostic);
npunlock_status npunlock_derive_act_dma_act_input_observations(
    const npunlock_invocation_routing_observation *invocations, size_t invocation_count,
    size_t group_count, const npunlock_dma_task_observation *dma_tasks, size_t dma_task_count,
    const npunlock_act_dma_source_candidate_observation *source_candidates,
    size_t source_candidate_count, npunlock_act_dma_act_input_candidate_observation **candidates,
    size_t *candidate_count, npunlock_act_dma_act_input_cover_observation **covers,
    size_t *cover_count, npunlock_diagnostic *diagnostic);
npunlock_status npunlock_derive_act_dma_dpu_input_observations(
    const npunlock_invocation_routing_observation *invocations, size_t invocation_count,
    size_t group_count, uint32_t tile_count,
    const npunlock_act_dma_source_candidate_observation *source_candidates,
    size_t source_candidate_count, const npunlock_act_dma_fanin_observation *fanin_observations,
    size_t fanin_observation_count, const npunlock_dpu_invariant_observation *dpu_invariants,
    size_t dpu_invariant_count, npunlock_act_dma_dpu_input_candidate_observation **candidates,
    size_t *candidate_count, npunlock_act_dma_dpu_input_cover_observation **covers,
    size_t *cover_count, npunlock_diagnostic *diagnostic);
npunlock_status npunlock_derive_act_consumer_cover_observations(
    const npunlock_invocation_routing_observation *invocations, size_t invocation_count,
    size_t group_count, const npunlock_act_edge_candidate_observation *candidates,
    size_t candidate_count, npunlock_act_consumer_cover_observation **covers, size_t *cover_count,
    npunlock_diagnostic *diagnostic);
npunlock_status npunlock_derive_act_output_cover_observations(
    const npunlock_invocation_routing_observation *invocations, size_t invocation_count,
    const npunlock_network_output_route_observation *routes, size_t route_count,
    const npunlock_act_output_slice_candidate_observation *slice_candidates,
    size_t slice_candidate_count, npunlock_act_output_cover_observation **covers,
    size_t *cover_count, npunlock_diagnostic *diagnostic);
npunlock_status npunlock_derive_act_replica_candidates(
    const npunlock_invocation_routing_observation *invocations, size_t invocation_count,
    size_t group_count, const npunlock_act_output_slice_candidate_observation *slice_candidates,
    size_t slice_candidate_count, npunlock_act_replica_group_observation **groups,
    size_t *replica_group_count, npunlock_act_replica_candidate_observation **candidates,
    size_t *replica_candidate_count, npunlock_diagnostic *diagnostic);
npunlock_status npunlock_derive_act_group_completeness(
    const npunlock_invocation_routing_observation *invocations, size_t invocation_count,
    size_t group_count, const npunlock_act_output_slice_candidate_observation *slice_candidates,
    size_t slice_candidate_count, const npunlock_network_output_route_observation *routes,
    size_t route_count, const npunlock_act_output_cover_observation *covers, size_t cover_count,
    const npunlock_act_replica_group_observation *replica_groups, size_t replica_group_count,
    const npunlock_act_replica_candidate_observation *replica_candidates,
    size_t replica_candidate_count, npunlock_act_group_completeness_observation **completeness,
    size_t *completeness_count, npunlock_diagnostic *diagnostic);
npunlock_status npunlock_derive_act_dpu_output_cover_observations(
    const npunlock_invocation_routing_observation *invocations, size_t invocation_count,
    size_t group_count, const npunlock_act_dpu_edge_candidate_observation *candidates,
    size_t candidate_count, npunlock_act_dpu_output_cover_observation **covers, size_t *cover_count,
    npunlock_diagnostic *diagnostic);
void npunlock_graph_routing_observations_release(npunlock_graph_routing_observations *observations);
npunlock_status
npunlock_compare_graph_routing_observations(const npunlock_graph_routing_observations *baseline,
                                            const npunlock_graph_routing_observations *candidate,
                                            npunlock_graph_routing_comparison *comparison,
                                            npunlock_diagnostic *diagnostic);
void npunlock_graph_routing_comparison_release(npunlock_graph_routing_comparison *comparison);

#endif
