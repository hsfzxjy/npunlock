#include <stdint.h>
#include <stdio.h>
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
  write_u64(graph + 0x28, 64);
  write_u16(graph + 0x3a, 64);
  write_u16(graph + 0x3c, 2);
  write_u16(graph + 0x3e, 1);
  write_u32(strings, 1);
  write_u32(strings + 4, 3);
  write_u64(strings + 0x18, 192);
  write_u64(strings + 0x20, sizeof(names));
  memcpy(graph + 192, names, sizeof(names));
}

int main(void) {
  uint8_t graph[201];
  const uint8_t image[] = {1, 2, 3, 4};
  patchblob_target target = {0};
  npunlock_buffer output = {0};
  npunlock_patch_summary summary = {0};
  npunlock_diagnostic diagnostic = {0};

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
