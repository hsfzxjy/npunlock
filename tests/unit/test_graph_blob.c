#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "graph_blob.h"
#include "npunlock/buffer.h"
#include "npunlock/error.h"

#define CHECK(expression)                                                                          \
  do {                                                                                             \
    if (!(expression)) {                                                                           \
      fprintf(stderr, "check failed at %s:%d: %s\n", __FILE__, __LINE__, #expression);             \
      return 1;                                                                                    \
    }                                                                                              \
  } while (0)

static void write_u16(uint8_t *bytes, uint16_t value) {
  bytes[0] = (uint8_t)value;
  bytes[1] = (uint8_t)(value >> 8);
}

static void write_u32(uint8_t *bytes, uint32_t value) {
  bytes[0] = (uint8_t)value;
  bytes[1] = (uint8_t)(value >> 8);
  bytes[2] = (uint8_t)(value >> 16);
  bytes[3] = (uint8_t)(value >> 24);
}

static void write_u64(uint8_t *bytes, uint64_t value) {
  write_u32(bytes, (uint32_t)value);
  write_u32(bytes + 4, (uint32_t)(value >> 32));
}

static void make_graph_without_act(uint8_t graph[201]) {
  static const uint8_t names[] = {0, '.', 's', 'h', 's', 't', 'r', 0, 0};
  uint8_t *strings = graph + 64u + 64u;
  memset(graph, 0, 201);
  memcpy(graph,
         "\x7f"
         "ELF",
         4);
  graph[4] = 2;
  graph[5] = 1;
  write_u16(graph + 0x10, 1);
  write_u64(graph + 0x28, 64);
  write_u16(graph + 0x34, 64);
  write_u16(graph + 0x3a, 64);
  write_u16(graph + 0x3c, 2);
  write_u16(graph + 0x3e, 1);
  write_u32(strings, 1);
  write_u32(strings + 4, 3);
  write_u64(strings + 0x18, 192);
  write_u64(strings + 0x20, sizeof(names));
  memcpy(graph + 192, names, sizeof(names));
}

static void configure_edge_invocation(npunlock_invocation_routing_observation *observation,
                                      uint32_t invocation_index, uint32_t group_index,
                                      int64_t input_addend, int64_t output_addend,
                                      uint64_t wait_mask, uint64_t post_mask) {
  patchblob_tensor_contract_v2 *input = &observation->target.tensors[0];
  patchblob_tensor_contract_v2 *output = &observation->target.tensors[1];
  memset(observation, 0, sizeof(*observation));
  observation->group_index = group_index;
  observation->wait_mask_raw = wait_mask;
  observation->post_mask_raw = post_mask;
  observation->target.invocation_index = invocation_index;
  observation->target.tensor_count = 2;
  input->role = PATCHBLOB_TENSOR_ROLE_INPUT_V2;
  input->precision = PATCHBLOB_TENSOR_PRECISION_FP16_V2;
  input->element_count = 8;
  input->span_bytes = 16;
  output->role = PATCHBLOB_TENSOR_ROLE_OUTPUT_V2;
  output->precision = PATCHBLOB_TENSOR_PRECISION_FP16_V2;
  output->element_count = 8;
  output->span_bytes = 16;
  observation->tensors[0].data_addend = input_addend;
  observation->tensors[1].data_addend = output_addend;
}

static int test_edge_candidate_ambiguity(void) {
  npunlock_invocation_routing_observation invocations[3];
  npunlock_act_edge_candidate_observation *candidates = NULL;
  size_t candidate_count = 0;
  npunlock_diagnostic diagnostic = {0};

  configure_edge_invocation(&invocations[0], 0, 0, 0, 64, 1, 2);
  configure_edge_invocation(&invocations[1], 1, 0, 16, 64, 1, 2);
  configure_edge_invocation(&invocations[2], 2, 1, 64, 96, 2, 4);
  CHECK(npunlock_derive_act_edge_candidates(invocations, 3, &candidates, &candidate_count,
                                            &diagnostic) == NPUNLOCK_STATUS_OK);
  CHECK(candidate_count == 2);
  CHECK(candidates[0].producer_invocation_index == 0);
  CHECK(candidates[1].producer_invocation_index == 1);
  CHECK(candidates[0].consumer_candidate_count == 2);
  CHECK(candidates[1].consumer_candidate_count == 2);
  CHECK(candidates[0].schedule_hop_count == 1);
  CHECK(candidates[1].schedule_hop_count == 1);
  free(candidates);
  candidates = NULL;

  invocations[1].tensors[1].data_addend = 80;
  CHECK(npunlock_derive_act_edge_candidates(invocations, 3, &candidates, &candidate_count,
                                            &diagnostic) == NPUNLOCK_STATUS_OK);
  CHECK(candidate_count == 1);
  CHECK(candidates[0].producer_invocation_index == 0);
  CHECK(candidates[0].consumer_candidate_count == 1);
  free(candidates);
  candidates = NULL;

  invocations[2].wait_mask_raw = 0;
  CHECK(npunlock_derive_act_edge_candidates(invocations, 3, &candidates, &candidate_count,
                                            &diagnostic) == NPUNLOCK_STATUS_OK);
  CHECK(candidate_count == 0);
  CHECK(candidates == NULL);
  npunlock_diagnostic_release(&diagnostic);
  return 0;
}

static int test_edge_candidate_transitive_schedule(void) {
  npunlock_invocation_routing_observation invocations[3];
  npunlock_act_edge_candidate_observation *candidates = NULL;
  size_t candidate_count = 0;
  npunlock_diagnostic diagnostic = {0};

  configure_edge_invocation(&invocations[0], 0, 0, 0, 64, 1, 2);
  configure_edge_invocation(&invocations[1], 1, 1, 80, 128, 2, 4);
  configure_edge_invocation(&invocations[2], 2, 2, 64, 192, 4, 8);
  CHECK(npunlock_derive_act_edge_candidates(invocations, 3, &candidates, &candidate_count,
                                            &diagnostic) == NPUNLOCK_STATUS_OK);
  CHECK(candidate_count == 1);
  CHECK(candidates[0].producer_invocation_index == 0);
  CHECK(candidates[0].consumer_invocation_index == 2);
  CHECK(candidates[0].schedule_hop_count == 2);
  CHECK(candidates[0].barrier_intersection_raw == 0);
  free(candidates);
  candidates = NULL;

  invocations[1].post_mask_raw = 0;
  CHECK(npunlock_derive_act_edge_candidates(invocations, 3, &candidates, &candidate_count,
                                            &diagnostic) == NPUNLOCK_STATUS_OK);
  CHECK(candidate_count == 0);
  CHECK(candidates == NULL);
  npunlock_diagnostic_release(&diagnostic);
  return 0;
}

static int test_output_cover_observations(void) {
  npunlock_invocation_routing_observation invocations[4];
  npunlock_network_output_route_observation routes[2] = {0};
  npunlock_act_output_slice_candidate_observation slices[4] = {0};
  npunlock_act_output_cover_observation *covers = NULL;
  size_t cover_count = 0;
  size_t index;
  npunlock_diagnostic diagnostic = {0};

  for (index = 0; index < 4; ++index) {
    configure_edge_invocation(&invocations[index], (uint32_t)index, (uint32_t)index, 0, 64, 1, 2);
    slices[index].act_group_index = (uint32_t)index;
    slices[index].act_invocation_index = (uint32_t)index;
    slices[index].route_candidate_count = 1;
    slices[index].output_symbol_raw = 1;
    slices[index].act_span_bytes = 16;
    slices[index].output_slice_addend_candidate = index * 16u;
  }
  routes[0].output_symbol_raw = 1;
  routes[0].output_addend_raw = 0;
  routes[0].length_bytes = 32;
  routes[1].output_symbol_raw = 1;
  routes[1].output_addend_raw = 32;
  routes[1].length_bytes = 32;

  CHECK(npunlock_derive_act_output_cover_observations(invocations, 4, routes, 2, slices, 4, &covers,
                                                      &cover_count,
                                                      &diagnostic) == NPUNLOCK_STATUS_OK);
  CHECK(cover_count == 1);
  CHECK(covers[0].exact_cover);
  CHECK(covers[0].unresolved_flags == 0);
  CHECK(covers[0].act_group_count == 4);
  CHECK(covers[0].group_invocation_count == 4);
  CHECK(covers[0].routed_invocation_count == 4);
  CHECK(covers[0].unrouted_invocation_count == 0);
  free(covers);
  covers = NULL;

  slices[3].output_slice_addend_candidate = 40;
  CHECK(npunlock_derive_act_output_cover_observations(invocations, 4, routes, 2, slices, 4, &covers,
                                                      &cover_count,
                                                      &diagnostic) == NPUNLOCK_STATUS_OK);
  CHECK(cover_count == 1);
  CHECK(!covers[0].exact_cover);
  CHECK((covers[0].unresolved_flags & NPUNLOCK_OUTPUT_COVER_SLICE_OVERLAP) != 0);
  CHECK((covers[0].unresolved_flags & NPUNLOCK_OUTPUT_COVER_UNION_MISMATCH) != 0);
  free(covers);
  covers = NULL;

  slices[3].output_slice_addend_candidate = 48;
  slices[0].route_candidate_count = 2;
  CHECK(npunlock_derive_act_output_cover_observations(invocations, 4, routes, 2, slices, 4, &covers,
                                                      &cover_count,
                                                      &diagnostic) == NPUNLOCK_STATUS_OK);
  CHECK(cover_count == 1);
  CHECK(!covers[0].exact_cover);
  CHECK((covers[0].unresolved_flags & NPUNLOCK_OUTPUT_COVER_ROUTE_AMBIGUITY) != 0);
  free(covers);
  npunlock_diagnostic_release(&diagnostic);
  return 0;
}

static int test_replica_candidate_bijection(void) {
  npunlock_invocation_routing_observation invocations[4];
  npunlock_act_output_slice_candidate_observation slices[2] = {0};
  npunlock_act_replica_group_observation *groups = NULL;
  npunlock_act_replica_candidate_observation *candidates = NULL;
  npunlock_network_output_route_observation route = {0};
  npunlock_act_group_completeness_observation *completeness = NULL;
  size_t group_count = 0;
  size_t candidate_count = 0;
  size_t completeness_count = 0;
  npunlock_diagnostic diagnostic = {0};

  configure_edge_invocation(&invocations[0], 0, 0, 0, 64, 1, 2);
  configure_edge_invocation(&invocations[1], 1, 0, 100, 164, 1, 2);
  configure_edge_invocation(&invocations[2], 2, 0, 16, 80, 1, 2);
  configure_edge_invocation(&invocations[3], 3, 0, 116, 180, 1, 2);
  invocations[0].tile_candidate_raw = 0;
  invocations[1].tile_candidate_raw = 1;
  invocations[2].tile_candidate_raw = 0;
  invocations[3].tile_candidate_raw = 1;
  slices[0].act_invocation_index = 0;
  slices[0].route_candidate_count = 1;
  slices[0].output_symbol_raw = 1;
  slices[0].act_span_bytes = 16;
  slices[1].act_invocation_index = 2;
  slices[1].route_candidate_count = 1;
  slices[1].output_symbol_raw = 1;
  slices[1].output_slice_addend_candidate = 16;
  slices[1].act_span_bytes = 16;
  route.output_symbol_raw = 1;
  route.length_bytes = 32;

  CHECK(npunlock_derive_act_replica_candidates(invocations, 4, 1, slices, 2, &groups, &group_count,
                                               &candidates, &candidate_count,
                                               &diagnostic) == NPUNLOCK_STATUS_OK);
  CHECK(group_count == 1);
  CHECK(groups[0].translation_solution_count == 1);
  CHECK(groups[0].cmx_translation_candidate == 100);
  CHECK(candidate_count == 2);
  CHECK(candidates[0].unrouted_invocation_index == 1);
  CHECK(candidates[0].routed_invocation_index == 0);
  CHECK(candidates[1].unrouted_invocation_index == 3);
  CHECK(candidates[1].routed_invocation_index == 2);
  CHECK(npunlock_derive_act_group_completeness(invocations, 4, 1, slices, 2, &route, 1, NULL, 0,
                                               groups, group_count, candidates, candidate_count,
                                               &completeness, &completeness_count,
                                               &diagnostic) == NPUNLOCK_STATUS_OK);
  CHECK(completeness_count == 1);
  CHECK(completeness[0].complete);
  CHECK(completeness[0].routed_invocation_count == 2);
  CHECK(completeness[0].replica_invocation_count == 2);
  free(completeness);
  completeness = NULL;
  free(groups);
  free(candidates);
  groups = NULL;
  candidates = NULL;

  invocations[3].tensors[0].data_addend = 100;
  invocations[3].tensors[1].data_addend = 164;
  CHECK(npunlock_derive_act_replica_candidates(invocations, 4, 1, slices, 2, &groups, &group_count,
                                               &candidates, &candidate_count,
                                               &diagnostic) == NPUNLOCK_STATUS_OK);
  CHECK(group_count == 1);
  CHECK(groups[0].translation_solution_count == 0);
  CHECK(candidate_count == 0);
  CHECK(npunlock_derive_act_group_completeness(invocations, 4, 1, slices, 2, &route, 1, NULL, 0,
                                               groups, group_count, candidates, candidate_count,
                                               &completeness, &completeness_count,
                                               &diagnostic) == NPUNLOCK_STATUS_OK);
  CHECK(completeness_count == 1);
  CHECK(!completeness[0].complete);
  CHECK(completeness[0].unresolved_invocation_count == 2);
  CHECK((completeness[0].unresolved_flags & NPUNLOCK_GROUP_COMPLETE_REPLICA_UNRESOLVED) != 0);
  free(completeness);
  free(groups);
  free(candidates);
  npunlock_diagnostic_release(&diagnostic);
  return 0;
}

static int test_group_local_output_cover_with_cmx_reuse(void) {
  npunlock_invocation_routing_observation invocations[3];
  npunlock_network_output_route_observation route = {0};
  npunlock_act_output_slice_candidate_observation slices[3] = {0};
  npunlock_act_output_cover_observation *global_covers = NULL;
  npunlock_act_group_completeness_observation *completeness = NULL;
  size_t global_cover_count = 0;
  size_t completeness_count = 0;
  size_t index;
  npunlock_diagnostic diagnostic = {0};

  configure_edge_invocation(&invocations[0], 0, 0, 0, 64, 1, 2);
  configure_edge_invocation(&invocations[1], 1, 0, 64, 128, 1, 2);
  configure_edge_invocation(&invocations[2], 2, 1, 64, 96, 1, 2);
  for (index = 0; index < 3; ++index) {
    slices[index].act_group_index = invocations[index].group_index;
    slices[index].act_invocation_index = (uint32_t)index;
    slices[index].route_candidate_count = 1;
    slices[index].output_symbol_raw = 7;
  }
  slices[0].act_span_bytes = 64;
  slices[1].output_slice_addend_candidate = 64;
  slices[1].act_span_bytes = 64;
  slices[2].output_slice_addend_candidate = 64;
  slices[2].act_span_bytes = 32;
  route.output_symbol_raw = 7;
  route.length_bytes = 128;

  CHECK(npunlock_derive_act_output_cover_observations(invocations, 3, &route, 1, slices, 3,
                                                      &global_covers, &global_cover_count,
                                                      &diagnostic) == NPUNLOCK_STATUS_OK);
  CHECK(global_cover_count == 1);
  CHECK(!global_covers[0].exact_cover);
  CHECK((global_covers[0].unresolved_flags & NPUNLOCK_OUTPUT_COVER_SLICE_OVERLAP) != 0);
  CHECK(npunlock_derive_act_group_completeness(
            invocations, 3, 2, slices, 3, &route, 1, global_covers, global_cover_count, NULL, 0,
            NULL, 0, &completeness, &completeness_count, &diagnostic) == NPUNLOCK_STATUS_OK);
  CHECK(completeness_count == 2);
  CHECK(completeness[0].act_group_index == 0);
  CHECK(completeness[0].complete);
  CHECK(completeness[1].act_group_index == 1);
  CHECK(!completeness[1].complete);
  CHECK((completeness[1].unresolved_flags & NPUNLOCK_GROUP_COMPLETE_OUTPUT_COVER_UNRESOLVED) != 0);

  free(global_covers);
  free(completeness);
  global_covers = NULL;
  completeness = NULL;

  invocations[1].group_index = 1;
  slices[1].act_group_index = 1;
  CHECK(npunlock_derive_act_output_cover_observations(invocations, 2, &route, 1, slices, 2,
                                                      &global_covers, &global_cover_count,
                                                      &diagnostic) == NPUNLOCK_STATUS_OK);
  CHECK(global_cover_count == 1);
  CHECK(global_covers[0].exact_cover);
  CHECK(npunlock_derive_act_group_completeness(
            invocations, 2, 2, slices, 2, &route, 1, global_covers, global_cover_count, NULL, 0,
            NULL, 0, &completeness, &completeness_count, &diagnostic) == NPUNLOCK_STATUS_OK);
  CHECK(completeness_count == 2);
  CHECK(completeness[0].complete);
  CHECK(completeness[1].complete);

  free(global_covers);
  free(completeness);
  npunlock_diagnostic_release(&diagnostic);
  return 0;
}

static int test_act_dpu_output_cover_observations(void) {
  npunlock_invocation_routing_observation invocations[4];
  npunlock_act_dpu_edge_candidate_observation candidates[4] = {{0}};
  npunlock_act_dpu_output_cover_observation *covers = NULL;
  size_t cover_count = 0;
  size_t index;
  npunlock_diagnostic diagnostic = {0};

  for (index = 0; index < 4; ++index) {
    uint32_t dpu_index = (uint32_t)(index & 1u);
    int64_t dpu_start = dpu_index == 0u ? 100 : 200;
    configure_edge_invocation(&invocations[index], (uint32_t)index, 0, 0, 0, 1, 2);
    candidates[index].direction = NPUNLOCK_ACT_DPU_DIRECTION_ACT_TO_DPU;
    candidates[index].act_group_index = 0;
    candidates[index].act_invocation_index = (uint32_t)index;
    candidates[index].act_tensor_record_index = 1;
    candidates[index].dpu_section_index = 14;
    candidates[index].dpu_invariant_index = dpu_index;
    candidates[index].act_candidate_count = 1;
    candidates[index].data_symbol_raw = 0;
    candidates[index].act_data_addend_raw = dpu_start + (index >= 2u ? 16 : 0);
    candidates[index].act_span_bytes = 16;
    candidates[index].dpu_data_addend_raw = dpu_start;
    candidates[index].dpu_span_bytes = 32;
  }

  CHECK(npunlock_derive_act_dpu_output_cover_observations(invocations, 4, 1, candidates, 4, &covers,
                                                          &cover_count,
                                                          &diagnostic) == NPUNLOCK_STATUS_OK);
  CHECK(cover_count == 1);
  CHECK(covers[0].exact_cover);
  CHECK(covers[0].invocation_count == 4);
  CHECK(covers[0].matched_invocation_count == 4);
  CHECK(covers[0].dpu_input_count == 2);
  CHECK(covers[0].exact_cover_input_count == 2);
  free(covers);
  covers = NULL;

  candidates[3].act_data_addend_raw = 208;
  CHECK(npunlock_derive_act_dpu_output_cover_observations(invocations, 4, 1, candidates, 4, &covers,
                                                          &cover_count,
                                                          &diagnostic) == NPUNLOCK_STATUS_OK);
  CHECK(cover_count == 1);
  CHECK(!covers[0].exact_cover);
  CHECK((covers[0].unresolved_flags & NPUNLOCK_ACT_DPU_COVER_SLICE_OVERLAP) != 0);
  CHECK((covers[0].unresolved_flags & NPUNLOCK_ACT_DPU_COVER_UNION_MISMATCH) != 0);
  free(covers);
  covers = NULL;

  candidates[3].act_data_addend_raw = 216;
  candidates[3].act_candidate_count = 2;
  CHECK(npunlock_derive_act_dpu_output_cover_observations(invocations, 4, 1, candidates, 4, &covers,
                                                          &cover_count,
                                                          &diagnostic) == NPUNLOCK_STATUS_OK);
  CHECK(cover_count == 1);
  CHECK(!covers[0].exact_cover);
  CHECK(covers[0].unresolved_invocation_count == 1);
  CHECK((covers[0].unresolved_flags & NPUNLOCK_ACT_DPU_COVER_CONSUMER_AMBIGUOUS) != 0);
  CHECK((covers[0].unresolved_flags & NPUNLOCK_ACT_DPU_COVER_INVOCATION_UNCLASSIFIED) != 0);
  free(covers);
  npunlock_diagnostic_release(&diagnostic);
  return 0;
}

static int test_act_consumer_cover_observations(void) {
  npunlock_invocation_routing_observation invocations[8];
  npunlock_act_edge_candidate_observation candidates[4] = {{0}};
  npunlock_act_consumer_cover_observation *covers = NULL;
  size_t cover_count = 0;
  size_t index;
  npunlock_diagnostic diagnostic = {0};

  for (index = 0; index < 4; ++index) {
    configure_edge_invocation(&invocations[index], (uint32_t)index, 0, 0, 0, 1, 2);
    configure_edge_invocation(&invocations[index + 4u], (uint32_t)(index + 4u), 1, 0, 0, 2, 4);
    candidates[index].producer_group_index = 0;
    candidates[index].producer_invocation_index = (uint32_t)index;
    candidates[index].producer_tensor_record_index = 1;
    candidates[index].consumer_group_index = 1;
    candidates[index].consumer_invocation_index = (uint32_t)(index + 4u);
    candidates[index].consumer_tensor_record_index = 0;
    candidates[index].consumer_candidate_count = 1;
  }

  CHECK(npunlock_derive_act_consumer_cover_observations(invocations, 8, 2, candidates, 4, &covers,
                                                        &cover_count,
                                                        &diagnostic) == NPUNLOCK_STATUS_OK);
  CHECK(cover_count == 1);
  CHECK(covers[0].exact_cover);
  CHECK(covers[0].producer_invocation_count == 4);
  CHECK(covers[0].matched_producer_invocation_count == 4);
  CHECK(covers[0].consumer_group_count == 1);
  CHECK(covers[0].consumer_input_count == 4);
  free(covers);
  covers = NULL;

  candidates[3].producer_invocation_index = 2;
  CHECK(npunlock_derive_act_consumer_cover_observations(invocations, 8, 2, candidates, 4, &covers,
                                                        &cover_count,
                                                        &diagnostic) == NPUNLOCK_STATUS_OK);
  CHECK(cover_count == 1);
  CHECK(!covers[0].exact_cover);
  CHECK(covers[0].unresolved_producer_invocation_count == 2);
  CHECK((covers[0].unresolved_flags & NPUNLOCK_ACT_CONSUMER_COVER_EDGE_AMBIGUOUS) != 0);
  CHECK((covers[0].unresolved_flags & NPUNLOCK_ACT_CONSUMER_COVER_PRODUCER_UNCLASSIFIED) != 0);
  free(covers);
  covers = NULL;

  candidates[3].producer_invocation_index = 3;
  candidates[3].consumer_invocation_index = 6;
  CHECK(npunlock_derive_act_consumer_cover_observations(invocations, 8, 2, candidates, 4, &covers,
                                                        &cover_count,
                                                        &diagnostic) == NPUNLOCK_STATUS_OK);
  CHECK(cover_count == 1);
  CHECK(!covers[0].exact_cover);
  CHECK((covers[0].unresolved_flags & NPUNLOCK_ACT_CONSUMER_COVER_INPUT_DUPLICATED) != 0);
  free(covers);
  npunlock_diagnostic_release(&diagnostic);
  return 0;
}

static void configure_contiguous_dma(npunlock_dma_task_observation *task, uint32_t section_index,
                                     uint32_t task_index, uint32_t engine_index,
                                     uint32_t length_bytes, uint64_t wait_mask,
                                     uint64_t post_mask) {
  memset(task, 0, sizeof(*task));
  task->section_index = section_index;
  task->task_index = task_index;
  task->engine_index = engine_index;
  task->transfer_type_raw = 1;
  task->length_bytes = length_bytes;
  task->source_width_raw = length_bytes;
  task->source_stride_raw = (int32_t)length_bytes;
  task->destination_width_raw = length_bytes;
  task->destination_stride_raw = (int32_t)length_bytes;
  task->wait_mask_raw = wait_mask;
  task->post_mask_raw = post_mask;
}

static int test_act_dma_fanin_observations(void) {
  npunlock_invocation_routing_observation invocations[2];
  npunlock_dma_task_observation dma_tasks[3];
  npunlock_act_dma_source_candidate_observation *candidates = NULL;
  npunlock_act_dma_fanin_observation *observations = NULL;
  npunlock_act_dma_dpu_input_candidate_observation *dpu_candidates = NULL;
  npunlock_act_dma_dpu_input_cover_observation *dpu_covers = NULL;
  npunlock_dpu_invariant_observation dpu_invariants[2] = {0};
  size_t candidate_count = 0;
  size_t observation_count = 0;
  size_t dpu_candidate_count = 0;
  size_t dpu_cover_count = 0;
  npunlock_diagnostic diagnostic = {0};

  configure_edge_invocation(&invocations[0], 8, 0, 128, 192, 16, 32);
  configure_edge_invocation(&invocations[1], 9, 0, 2097280, 2097344, 16, 32);
  invocations[1].tile_candidate_raw = 1;
  invocations[0].target.tensors[1].element_count = 16;
  invocations[0].target.tensors[1].span_bytes = 32;
  invocations[1].target.tensors[1].element_count = 16;
  invocations[1].target.tensors[1].span_bytes = 32;

  configure_contiguous_dma(&dma_tasks[0], 5, 8, 0, 32, 32, 64);
  dma_tasks[0].source.memory_kind = NPUNLOCK_DMA_ENDPOINT_CMX;
  dma_tasks[0].source.relocation_type = 0;
  dma_tasks[0].source.symbol_raw = 0;
  dma_tasks[0].source.addend_raw = 192;
  dma_tasks[0].destination.memory_kind = NPUNLOCK_DMA_ENDPOINT_DDR_BUFFER;
  dma_tasks[0].destination.symbol_table_index = 26;
  dma_tasks[0].destination.symbol_raw = 1;
  dma_tasks[0].destination.addend_raw = 0;

  configure_contiguous_dma(&dma_tasks[1], 6, 6, 1, 32, 32, 64);
  dma_tasks[1].source.memory_kind = NPUNLOCK_DMA_ENDPOINT_CMX;
  dma_tasks[1].source.relocation_type = 0;
  dma_tasks[1].source.symbol_raw = 0;
  dma_tasks[1].source.addend_raw = 2097344;
  dma_tasks[1].destination.memory_kind = NPUNLOCK_DMA_ENDPOINT_DDR_BUFFER;
  dma_tasks[1].destination.symbol_table_index = 26;
  dma_tasks[1].destination.symbol_raw = 1;
  dma_tasks[1].destination.addend_raw = 32;

  configure_contiguous_dma(&dma_tasks[2], 5, 9, 0, 64, 64, 128);
  dma_tasks[2].source.memory_kind = NPUNLOCK_DMA_ENDPOINT_DDR_BUFFER;
  dma_tasks[2].source.relocation_type = 0;
  dma_tasks[2].source.symbol_table_index = 26;
  dma_tasks[2].source.symbol_raw = 1;
  dma_tasks[2].source.addend_raw = 0;
  dma_tasks[2].destination.memory_kind = NPUNLOCK_DMA_ENDPOINT_CMX;
  dma_tasks[2].destination.relocation_type = 1;
  dma_tasks[2].destination.addend_raw = 128;
  dma_tasks[2].destination_value_raw = UINT64_C(0xc00000);

  CHECK(npunlock_derive_act_dma_source_candidates(invocations, 2, dma_tasks, 3, &candidates,
                                                  &candidate_count,
                                                  &diagnostic) == NPUNLOCK_STATUS_OK);
  CHECK(candidate_count == 2);
  CHECK(candidates[0].dma_candidate_count == 1);
  CHECK(candidates[0].dma_task_index == 8);
  CHECK(candidates[0].ddr_destination_addend_raw == 0);
  CHECK(candidates[0].dma_post_mask_raw == 64);
  CHECK(candidates[1].dma_candidate_count == 1);
  CHECK(candidates[1].dma_engine_index == 1);
  CHECK(candidates[1].ddr_destination_addend_raw == 32);

  CHECK(npunlock_derive_act_dma_fanin_observations(
            invocations, 2, 1, dma_tasks, 3, candidates, candidate_count, &observations,
            &observation_count, &diagnostic) == NPUNLOCK_STATUS_OK);
  CHECK(observation_count == 1);
  CHECK(observations[0].exact_cover);
  CHECK(observations[0].unresolved_flags == 0);
  CHECK(observations[0].matched_invocation_count == 2);
  CHECK(observations[0].source_dma_task_count == 2);
  CHECK(observations[0].fanin_candidate_count == 1);
  CHECK(observations[0].fanin_dma_task_index == 9);
  CHECK(observations[0].span_bytes == 64);
  CHECK(observations[0].cmx_destination_value_raw == UINT64_C(0xc00000));

  dpu_invariants[0].section_index = 14;
  dpu_invariants[0].invariant_index = 0;
  dpu_invariants[0].cluster_raw = 0;
  dpu_invariants[0].wait_mask_raw = 128;
  dpu_invariants[0].input_span_bytes = 64;
  dpu_invariants[0].input_count = 2;
  dpu_invariants[0].inputs[0].field_index = 0;
  dpu_invariants[0].inputs[0].data_symbol_raw = 0;
  dpu_invariants[0].inputs[0].data_addend_raw = 128;
  dpu_invariants[0].inputs[1].field_index = 1;
  dpu_invariants[0].inputs[1].data_symbol_raw = 0;
  dpu_invariants[0].inputs[1].data_addend_raw = 128;
  dpu_invariants[1] = dpu_invariants[0];
  dpu_invariants[1].invariant_index = 1;
  dpu_invariants[1].cluster_raw = 1;
  dpu_invariants[1].inputs[0].data_addend_raw = 2097280;
  dpu_invariants[1].inputs[1].data_addend_raw = 2097280;

  CHECK(npunlock_derive_act_dma_dpu_input_observations(
            invocations, 2, 1, 2, candidates, candidate_count, observations, observation_count,
            dpu_invariants, 2, &dpu_candidates, &dpu_candidate_count, &dpu_covers, &dpu_cover_count,
            &diagnostic) == NPUNLOCK_STATUS_OK);
  CHECK(dpu_candidate_count == 2);
  CHECK(dpu_cover_count == 1);
  CHECK(dpu_covers[0].exact_cover);
  CHECK(dpu_covers[0].translated_tile_count == 2);
  CHECK(dpu_covers[0].matched_dpu_input_count == 2);
  CHECK(dpu_candidates[0].tile_translation_candidate == 0);
  CHECK(dpu_candidates[0].dpu_field_mask_raw == 3);
  CHECK(dpu_candidates[1].tile_translation_candidate == 2097152);
  CHECK(dpu_candidates[1].translated_destination_addend_raw == 2097280);
  CHECK(dpu_candidates[1].dpu_input_addend_raw == 2097280);
  CHECK(dpu_candidates[1].barrier_intersection_raw == 128);
  free(dpu_candidates);
  free(dpu_covers);
  dpu_candidates = NULL;
  dpu_covers = NULL;

  dpu_invariants[0].input_ambiguous_fields_raw = 1;
  dpu_invariants[1].input_ambiguous_fields_raw = 1;
  CHECK(npunlock_derive_act_dma_dpu_input_observations(
            invocations, 2, 1, 2, candidates, candidate_count, observations, observation_count,
            dpu_invariants, 2, &dpu_candidates, &dpu_candidate_count, &dpu_covers, &dpu_cover_count,
            &diagnostic) == NPUNLOCK_STATUS_OK);
  CHECK(dpu_candidate_count == 2);
  CHECK(dpu_covers[0].exact_cover);
  CHECK(dpu_candidates[0].dpu_field_mask_raw == 2);
  CHECK(dpu_candidates[1].dpu_field_mask_raw == 2);
  free(dpu_candidates);
  free(dpu_covers);
  dpu_candidates = NULL;
  dpu_covers = NULL;

  dpu_invariants[0].input_ambiguous_fields_raw = 3;
  dpu_invariants[1].input_ambiguous_fields_raw = 3;
  CHECK(npunlock_derive_act_dma_dpu_input_observations(
            invocations, 2, 1, 2, candidates, candidate_count, observations, observation_count,
            dpu_invariants, 2, &dpu_candidates, &dpu_candidate_count, &dpu_covers, &dpu_cover_count,
            &diagnostic) == NPUNLOCK_STATUS_OK);
  CHECK(dpu_candidate_count == 2);
  CHECK(!dpu_covers[0].exact_cover);
  CHECK(dpu_covers[0].matched_dpu_input_count == 0);
  CHECK((dpu_covers[0].unresolved_flags & NPUNLOCK_ACT_DMA_DPU_INPUT_AMBIGUOUS) != 0);
  free(dpu_candidates);
  free(dpu_covers);
  dpu_candidates = NULL;
  dpu_covers = NULL;

  dpu_invariants[0].input_ambiguous_fields_raw = 0;
  dpu_invariants[1].input_ambiguous_fields_raw = 0;

  dpu_invariants[1].wait_mask_raw = 0;
  CHECK(npunlock_derive_act_dma_dpu_input_observations(
            invocations, 2, 1, 2, candidates, candidate_count, observations, observation_count,
            dpu_invariants, 2, &dpu_candidates, &dpu_candidate_count, &dpu_covers, &dpu_cover_count,
            &diagnostic) == NPUNLOCK_STATUS_OK);
  CHECK(dpu_candidate_count == 2);
  CHECK(!dpu_covers[0].exact_cover);
  CHECK(dpu_covers[0].matched_dpu_input_count == 1);
  CHECK((dpu_covers[0].unresolved_flags & NPUNLOCK_ACT_DMA_DPU_INPUT_AMBIGUOUS) != 0);
  free(dpu_candidates);
  free(dpu_covers);
  free(observations);
  observations = NULL;

  dma_tasks[2].wait_mask_raw = 0;
  CHECK(npunlock_derive_act_dma_fanin_observations(
            invocations, 2, 1, dma_tasks, 3, candidates, candidate_count, &observations,
            &observation_count, &diagnostic) == NPUNLOCK_STATUS_OK);
  CHECK(observation_count == 1);
  CHECK(!observations[0].exact_cover);
  CHECK(observations[0].fanin_candidate_count == 0);
  CHECK((observations[0].unresolved_flags & NPUNLOCK_ACT_DMA_FANIN_TASK_AMBIGUOUS) != 0);
  free(observations);
  observations = NULL;

  dma_tasks[2].wait_mask_raw = 64;
  candidates[1].ddr_destination_addend_raw = 16;
  CHECK(npunlock_derive_act_dma_fanin_observations(
            invocations, 2, 1, dma_tasks, 3, candidates, candidate_count, &observations,
            &observation_count, &diagnostic) == NPUNLOCK_STATUS_OK);
  CHECK(observation_count == 1);
  CHECK(!observations[0].exact_cover);
  CHECK((observations[0].unresolved_flags & NPUNLOCK_ACT_DMA_FANIN_DDR_COVER_UNRESOLVED) != 0);
  free(observations);
  free(candidates);
  npunlock_diagnostic_release(&diagnostic);
  return 0;
}

static int test_act_dma_act_input_observations(void) {
  npunlock_invocation_routing_observation invocations[4];
  npunlock_dma_task_observation reloads[2];
  npunlock_act_dma_source_candidate_observation source = {0};
  npunlock_act_dma_act_input_candidate_observation *candidates = NULL;
  npunlock_act_dma_act_input_cover_observation *covers = NULL;
  size_t candidate_count = 0;
  size_t cover_count = 0;
  npunlock_diagnostic diagnostic = {0};

  configure_edge_invocation(&invocations[0], 0, 0, 0, 128, 1, 8);
  configure_edge_invocation(&invocations[1], 1, 0, 2097152, 2097280, 1, 8);
  invocations[1].tile_candidate_raw = 1;
  invocations[0].target.tensors[1].element_count = 32;
  invocations[0].target.tensors[1].span_bytes = 64;
  invocations[1].target.tensors[1].element_count = 32;
  invocations[1].target.tensors[1].span_bytes = 64;

  configure_edge_invocation(&invocations[2], 2, 1, 0, 64, 2, 16);
  configure_edge_invocation(&invocations[3], 3, 1, 2097152, 2097216, 2, 16);
  invocations[3].tile_candidate_raw = 1;
  invocations[2].target.tensors[0].element_count = 16;
  invocations[2].target.tensors[0].span_bytes = 32;
  invocations[3].target.tensors[0].element_count = 16;
  invocations[3].target.tensors[0].span_bytes = 32;

  source.act_group_index = 0;
  source.act_invocation_index = 0;
  source.act_tensor_record_index = 1;
  source.dma_section_index = 5;
  source.dma_task_index = 2;
  source.dma_candidate_count = 1;
  source.ddr_symbol_table_index = 26;
  source.ddr_symbol_raw = 1;
  source.ddr_destination_addend_raw = 0;
  source.span_bytes = 64;
  source.dma_post_mask_raw = 4;

  configure_contiguous_dma(&reloads[0], 5, 3, 0, 32, 4, 2);
  reloads[0].source.memory_kind = NPUNLOCK_DMA_ENDPOINT_DDR_BUFFER;
  reloads[0].source.symbol_table_index = 26;
  reloads[0].source.symbol_raw = 1;
  reloads[0].destination.memory_kind = NPUNLOCK_DMA_ENDPOINT_CMX;
  reloads[0].destination.symbol_raw = 0;
  configure_contiguous_dma(&reloads[1], 6, 1, 1, 32, 4, 2);
  reloads[1].source.memory_kind = NPUNLOCK_DMA_ENDPOINT_DDR_BUFFER;
  reloads[1].source.symbol_table_index = 26;
  reloads[1].source.symbol_raw = 1;
  reloads[1].source.addend_raw = 32;
  reloads[1].destination.memory_kind = NPUNLOCK_DMA_ENDPOINT_CMX;
  reloads[1].destination.symbol_raw = 0;
  reloads[1].destination.addend_raw = 2097152;

  CHECK(npunlock_derive_act_dma_act_input_observations(
            invocations, 4, 2, reloads, 2, &source, 1, &candidates, &candidate_count, &covers,
            &cover_count, &diagnostic) == NPUNLOCK_STATUS_OK);
  CHECK(candidate_count == 2);
  CHECK(cover_count == 1);
  CHECK(covers[0].exact_cover);
  CHECK(covers[0].routed_producer_invocation_count == 1);
  CHECK(covers[0].replica_producer_invocation_count == 1);
  CHECK(covers[0].consumer_group_count == 1);
  CHECK(covers[0].consumer_input_count == 2);
  CHECK(covers[0].ddr_span_bytes == 64);
  CHECK(candidates[0].consumer_candidate_count == 1);
  CHECK(candidates[1].ddr_source_addend_raw == 32);
  free(candidates);
  free(covers);
  candidates = NULL;
  covers = NULL;

  reloads[1].wait_mask_raw = 0;
  CHECK(npunlock_derive_act_dma_act_input_observations(
            invocations, 4, 2, reloads, 2, &source, 1, &candidates, &candidate_count, &covers,
            &cover_count, &diagnostic) == NPUNLOCK_STATUS_OK);
  CHECK(candidate_count == 1);
  CHECK(cover_count == 1);
  CHECK(!covers[0].exact_cover);
  CHECK((covers[0].unresolved_flags & NPUNLOCK_ACT_DMA_ACT_DDR_COVER_UNRESOLVED) != 0);
  free(candidates);
  free(covers);
  npunlock_diagnostic_release(&diagnostic);
  return 0;
}

static int test_routing_comparison(void) {
  npunlock_invocation_routing_observation baseline_invocations[2];
  npunlock_invocation_routing_observation candidate_invocations[2];
  npunlock_dma_task_observation baseline_dma = {0};
  npunlock_dma_task_observation candidate_dma = {0};
  npunlock_act_dma_source_candidate_observation baseline_act_dma = {0};
  npunlock_act_dma_source_candidate_observation candidate_act_dma = {0};
  npunlock_act_dma_dpu_input_candidate_observation baseline_act_dma_dpu = {0};
  npunlock_act_dma_dpu_input_candidate_observation candidate_act_dma_dpu = {0};
  npunlock_act_dma_act_input_candidate_observation baseline_act_dma_act = {0};
  npunlock_act_dma_act_input_candidate_observation candidate_act_dma_act = {0};
  npunlock_act_dma_act_input_cover_observation baseline_act_dma_act_cover = {0};
  npunlock_act_dma_act_input_cover_observation candidate_act_dma_act_cover = {0};
  npunlock_graph_routing_observations baseline = {0};
  npunlock_graph_routing_observations candidate = {0};
  npunlock_graph_routing_comparison comparison = {0};
  npunlock_diagnostic diagnostic = {0};

  configure_edge_invocation(&baseline_invocations[0], 0, 0, 0, 16, 1, 2);
  configure_edge_invocation(&baseline_invocations[1], 1, 1, 16, 32, 2, 4);
  baseline_invocations[0].code_extent = 16;
  baseline_invocations[0].code_sha256[0] = 1;
  baseline_invocations[1].code_extent = 16;
  baseline_invocations[1].code_sha256[0] = 2;
  memcpy(candidate_invocations, baseline_invocations, sizeof(candidate_invocations));
  candidate_invocations[1].code_sha256[0] = 3;
  baseline.invocations = baseline_invocations;
  baseline.invocation_count = 2;
  baseline.group_count = 2;
  baseline.tile_count_observed = 2;
  baseline.dma_tasks = &baseline_dma;
  baseline.dma_task_count = 1;
  baseline.act_dma_source_candidates = &baseline_act_dma;
  baseline.act_dma_source_candidate_count = 1;
  baseline.act_dma_dpu_input_candidates = &baseline_act_dma_dpu;
  baseline.act_dma_dpu_input_candidate_count = 1;
  baseline.act_dma_act_input_candidates = &baseline_act_dma_act;
  baseline.act_dma_act_input_candidate_count = 1;
  baseline.act_dma_act_input_covers = &baseline_act_dma_act_cover;
  baseline.act_dma_act_input_cover_count = 1;
  candidate.invocations = candidate_invocations;
  candidate.invocation_count = 2;
  candidate.group_count = 2;
  candidate.tile_count_observed = 2;
  candidate.dma_tasks = &candidate_dma;
  candidate.dma_task_count = 1;
  candidate.act_dma_source_candidates = &candidate_act_dma;
  candidate.act_dma_source_candidate_count = 1;
  candidate.act_dma_dpu_input_candidates = &candidate_act_dma_dpu;
  candidate.act_dma_dpu_input_candidate_count = 1;
  candidate.act_dma_act_input_candidates = &candidate_act_dma_act;
  candidate.act_dma_act_input_candidate_count = 1;
  candidate.act_dma_act_input_covers = &candidate_act_dma_act_cover;
  candidate.act_dma_act_input_cover_count = 1;

  CHECK(npunlock_compare_graph_routing_observations(&baseline, &candidate, &comparison,
                                                    &diagnostic) == NPUNLOCK_STATUS_OK);
  CHECK(comparison.contract_stable);
  CHECK(comparison.unique_changed_group);
  CHECK(comparison.changed_group_count == 1);
  CHECK(comparison.changed_group_indices[0] == 1);
  npunlock_graph_routing_comparison_release(&comparison);

  candidate_dma.length_bytes = 32;
  CHECK(npunlock_compare_graph_routing_observations(&baseline, &candidate, &comparison,
                                                    &diagnostic) == NPUNLOCK_STATUS_OK);
  CHECK(!comparison.contract_stable);
  CHECK((comparison.mismatch_flags & NPUNLOCK_COMPARE_DMA_CHANGED) != 0);
  npunlock_graph_routing_comparison_release(&comparison);
  candidate_dma.length_bytes = 0;

  candidate_act_dma.span_bytes = 32;
  CHECK(npunlock_compare_graph_routing_observations(&baseline, &candidate, &comparison,
                                                    &diagnostic) == NPUNLOCK_STATUS_OK);
  CHECK(!comparison.contract_stable);
  CHECK((comparison.mismatch_flags & NPUNLOCK_COMPARE_ACT_DMA_RELATIONS_CHANGED) != 0);
  npunlock_graph_routing_comparison_release(&comparison);
  candidate_act_dma.span_bytes = 0;

  candidate_act_dma_dpu.tile_translation_candidate = 2097152;
  CHECK(npunlock_compare_graph_routing_observations(&baseline, &candidate, &comparison,
                                                    &diagnostic) == NPUNLOCK_STATUS_OK);
  CHECK(!comparison.contract_stable);
  CHECK((comparison.mismatch_flags & NPUNLOCK_COMPARE_ACT_DMA_DPU_RELATIONS_CHANGED) != 0);
  npunlock_graph_routing_comparison_release(&comparison);
  candidate_act_dma_dpu.tile_translation_candidate = 0;

  candidate_act_dma_act.span_bytes = 32;
  CHECK(npunlock_compare_graph_routing_observations(&baseline, &candidate, &comparison,
                                                    &diagnostic) == NPUNLOCK_STATUS_OK);
  CHECK(!comparison.contract_stable);
  CHECK((comparison.mismatch_flags & NPUNLOCK_COMPARE_ACT_DMA_ACT_RELATIONS_CHANGED) != 0);
  npunlock_graph_routing_comparison_release(&comparison);
  candidate_act_dma_act.span_bytes = 0;

  candidate_act_dma_act_cover.consumer_input_count = 1;
  CHECK(npunlock_compare_graph_routing_observations(&baseline, &candidate, &comparison,
                                                    &diagnostic) == NPUNLOCK_STATUS_OK);
  CHECK(!comparison.contract_stable);
  CHECK((comparison.mismatch_flags & NPUNLOCK_COMPARE_ACT_DMA_ACT_RELATIONS_CHANGED) != 0);
  npunlock_graph_routing_comparison_release(&comparison);
  candidate_act_dma_act_cover.consumer_input_count = 0;

  candidate_invocations[1].wait_mask_raw = 8;
  CHECK(npunlock_compare_graph_routing_observations(&baseline, &candidate, &comparison,
                                                    &diagnostic) == NPUNLOCK_STATUS_OK);
  CHECK(!comparison.contract_stable);
  CHECK((comparison.mismatch_flags & NPUNLOCK_COMPARE_INVOCATION_CONTRACT_CHANGED) != 0);
  npunlock_graph_routing_comparison_release(&comparison);
  candidate_invocations[1].wait_mask_raw = baseline_invocations[1].wait_mask_raw;

  baseline.group_count = 1;
  candidate.group_count = 1;
  baseline_invocations[1].group_index = 0;
  candidate_invocations[1].group_index = 0;
  candidate_invocations[0].code_sha256[0] = 4;
  candidate_invocations[1].code_sha256[0] = baseline_invocations[1].code_sha256[0];
  CHECK(npunlock_compare_graph_routing_observations(&baseline, &candidate, &comparison,
                                                    &diagnostic) == NPUNLOCK_STATUS_OK);
  CHECK(!comparison.contract_stable);
  CHECK(comparison.unique_changed_group);
  CHECK((comparison.mismatch_flags & NPUNLOCK_COMPARE_PARTIAL_GROUP_CODE_CHANGE) != 0);
  npunlock_graph_routing_comparison_release(&comparison);

  baseline.invocations = NULL;
  baseline.invocation_count = 1;
  CHECK(npunlock_compare_graph_routing_observations(
            &baseline, &candidate, &comparison, &diagnostic) == NPUNLOCK_STATUS_INVALID_ARGUMENT);
  npunlock_diagnostic_release(&diagnostic);
  return 0;
}

int main(void) {
  uint8_t graph[201];
  const uint8_t image[] = {1, 2, 3, 4};
  patchblob_target target = {0};
  npunlock_buffer output = {0};
  npunlock_patch_summary summary = {0};
  npunlock_diagnostic diagnostic = {0};

  CHECK(test_edge_candidate_ambiguity() == 0);
  CHECK(test_edge_candidate_transitive_schedule() == 0);
  CHECK(test_output_cover_observations() == 0);
  CHECK(test_replica_candidate_bijection() == 0);
  CHECK(test_group_local_output_cover_with_cmx_reuse() == 0);
  CHECK(test_act_dpu_output_cover_observations() == 0);
  CHECK(test_act_consumer_cover_observations() == 0);
  CHECK(test_act_dma_fanin_observations() == 0);
  CHECK(test_act_dma_act_input_observations() == 0);
  CHECK(test_routing_comparison() == 0);

  target.struct_size = sizeof(target);
  target.invocation_index = 0;
  target.range_index = 0;
  target.expected_input_count = 1;
  target.expected_element_count = 1;
  target.expected_span_bytes = 2;
  target.required_contract_flags = PATCHBLOB_CONTRACT_STATIC | PATCHBLOB_CONTRACT_DENSE |
                                   PATCHBLOB_CONTRACT_FP16 | PATCHBLOB_CONTRACT_CMX |
                                   PATCHBLOB_CONTRACT_DISJOINT_OUTPUT;

  CHECK(npunlock_patch_graph_blob(
            (npunlock_view){image, sizeof(image)}, (npunlock_view){image, sizeof(image)}, &target,
            1, 0x400, 0x80, &output, &summary, &diagnostic) == NPUNLOCK_STATUS_MALFORMED_INPUT);
  npunlock_diagnostic_release(&diagnostic);

  make_graph_without_act(graph);
  CHECK(npunlock_patch_graph_blob((npunlock_view){graph, sizeof(graph)},
                                  (npunlock_view){image, sizeof(image)}, &target, 1, 0x400, 0x80,
                                  &output, &summary, &diagnostic) == NPUNLOCK_STATUS_UNSUPPORTED);
  CHECK(output.data == NULL && summary.details == NULL);
  npunlock_diagnostic_release(&diagnostic);

  write_u64(graph + 64u + 64u + 0x18u, UINT64_MAX);
  CHECK(npunlock_patch_graph_blob(
            (npunlock_view){graph, sizeof(graph)}, (npunlock_view){image, sizeof(image)}, &target,
            1, 0x400, 0x80, &output, &summary, &diagnostic) == NPUNLOCK_STATUS_MALFORMED_INPUT);
  npunlock_diagnostic_release(&diagnostic);
  return 0;
}
