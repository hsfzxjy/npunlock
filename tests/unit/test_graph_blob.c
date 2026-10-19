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

int main(void) {
  uint8_t graph[201];
  const uint8_t image[] = {1, 2, 3, 4};
  patchblob_target target = {0};
  npunlock_buffer output = {0};
  npunlock_patch_summary summary = {0};
  npunlock_diagnostic diagnostic = {0};

  CHECK(test_edge_candidate_ambiguity() == 0);

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
