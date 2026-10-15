#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "npunlock/patchblob.h"

#define FIXTURE_ROOT "tests/fixtures/npu3720/"

#define CHECK(expression)                                                                          \
  do {                                                                                             \
    if (!(expression)) {                                                                           \
      fprintf(stderr, "check failed at %s:%d: %s\n", __FILE__, __LINE__, #expression);             \
      return 1;                                                                                    \
    }                                                                                              \
  } while (0)

static npunlock_buffer read_file(const char *path) {
  npunlock_buffer buffer = {0};
  FILE *stream = NULL;
  long length;
#ifdef _WIN32
  if (fopen_s(&stream, path, "rb") != 0) {
    stream = NULL;
  }
#else
  stream = fopen(path, "rb");
#endif
  if (stream == NULL || fseek(stream, 0, SEEK_END) != 0) {
    if (stream != NULL) {
      fclose(stream);
    }
    return buffer;
  }
  length = ftell(stream);
  if (length <= 0 || fseek(stream, 0, SEEK_SET) != 0) {
    fclose(stream);
    return buffer;
  }
  buffer.data = (uint8_t *)malloc((size_t)length);
  if (buffer.data == NULL || fread(buffer.data, 1, (size_t)length, stream) != (size_t)length) {
    free(buffer.data);
    buffer.data = NULL;
    fclose(stream);
    return buffer;
  }
  buffer.size = (size_t)length;
  fclose(stream);
  return buffer;
}

static uint8_t *find_bytes(uint8_t *data, size_t data_size, const uint8_t *expected,
                           size_t expected_size) {
  size_t index;
  if (expected_size > data_size) {
    return NULL;
  }
  for (index = 0; index <= data_size - expected_size; ++index) {
    if (memcmp(data + index, expected, expected_size) == 0) {
      return data + index;
    }
  }
  return NULL;
}

static uint16_t read_u16(const uint8_t *bytes) {
  return (uint16_t)((uint16_t)bytes[0] | ((uint16_t)bytes[1] << 8));
}

static uint32_t read_u32(const uint8_t *bytes) {
  return (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) | ((uint32_t)bytes[2] << 16) |
         ((uint32_t)bytes[3] << 24);
}

static uint64_t read_u64(const uint8_t *bytes) {
  return (uint64_t)read_u32(bytes) | ((uint64_t)read_u32(bytes + 4) << 32);
}

static void write_u32(uint8_t *bytes, uint32_t value) {
  bytes[0] = (uint8_t)value;
  bytes[1] = (uint8_t)(value >> 8);
  bytes[2] = (uint8_t)(value >> 16);
  bytes[3] = (uint8_t)(value >> 24);
}

static void write_u64(uint8_t *bytes, uint64_t value) {
  bytes[0] = (uint8_t)value;
  bytes[1] = (uint8_t)(value >> 8);
  bytes[2] = (uint8_t)(value >> 16);
  bytes[3] = (uint8_t)(value >> 24);
  bytes[4] = (uint8_t)(value >> 32);
  bytes[5] = (uint8_t)(value >> 40);
  bytes[6] = (uint8_t)(value >> 48);
  bytes[7] = (uint8_t)(value >> 56);
}

static uint8_t *find_section_header(uint8_t *data, size_t size, const char *expected_name,
                                    uint32_t expected_link) {
  size_t section_table;
  size_t section_count;
  size_t string_index;
  size_t string_header;
  size_t string_offset;
  size_t string_size;
  size_t index;
  if (size < 64u || read_u64(data + 0x28u) > SIZE_MAX) {
    return NULL;
  }
  section_table = (size_t)read_u64(data + 0x28u);
  section_count = read_u16(data + 0x3cu);
  string_index = read_u16(data + 0x3eu);
  if (section_count == 0u || section_count > size / 64u || string_index >= section_count ||
      section_table > size - 64u * section_count) {
    return NULL;
  }
  string_header = section_table + 64u * string_index;
  if (read_u64(data + string_header + 0x18u) > SIZE_MAX ||
      read_u64(data + string_header + 0x20u) > SIZE_MAX) {
    return NULL;
  }
  string_offset = (size_t)read_u64(data + string_header + 0x18u);
  string_size = (size_t)read_u64(data + string_header + 0x20u);
  if (string_offset > size || string_size > size - string_offset) {
    return NULL;
  }
  for (index = 0; index < section_count; ++index) {
    uint8_t *header = data + section_table + 64u * index;
    uint32_t name_offset = read_u32(header);
    const char *name;
    if (name_offset >= string_size) {
      return NULL;
    }
    name = (const char *)(data + string_offset + name_offset);
    if (memchr(name, 0, string_size - name_offset) == NULL) {
      return NULL;
    }
    if (strcmp(name, expected_name) == 0 &&
        (expected_link == UINT32_MAX || read_u32(header + 0x28u) == expected_link)) {
      return header;
    }
  }
  return NULL;
}

static int sections_are_aligned(const uint8_t *data, size_t size) {
  size_t section_table;
  size_t section_count;
  size_t index;
  if (size < 64u || read_u64(data + 0x28u) > SIZE_MAX) {
    return 0;
  }
  section_table = (size_t)read_u64(data + 0x28u);
  section_count = read_u16(data + 0x3cu);
  if (section_count == 0u || section_count > size / 64u ||
      section_table > size - 64u * section_count) {
    return 0;
  }
  for (index = 0; index < section_count; ++index) {
    const uint8_t *header = data + section_table + 64u * index;
    uint64_t offset = read_u64(header + 0x18u);
    uint64_t alignment = read_u64(header + 0x30u);
    if (alignment > 1u && offset % alignment != 0u) {
      return 0;
    }
  }
  return 1;
}

int main(void) {
  npunlock_buffer carrier;
  npunlock_buffer two_operation_carrier;
  npunlock_buffer broadcast_carrier;
  npunlock_buffer convert_f32_to_f16;
  npunlock_buffer convert_f16_to_f32;
  npunlock_buffer i32_carrier;
  npunlock_buffer i8_carrier;
  npunlock_buffer elf;
  npunlock_buffer expected;
  npunlock_buffer unsupported_header = {0};
  npunlock_buffer unsupported_carrier = {0};
  npunlock_buffer unsupported_tiles = {0};
  npunlock_buffer unsupported_platform = {0};
  npunlock_buffer malformed_sections = {0};
  patchblob_options options = {0};
  patchblob_target targets[2] = {{0}};
  patchblob_target broadcast_targets[4] = {{0}};
  patchblob_target_v2 conversion_targets[2] = {{0}};
  patchblob_discovery_result discovery = {0};
  patchblob_discovery_result_v2 discovery_v2 = {0};
  patchblob_result result = {0};
  uint32_t flags = PATCHBLOB_CONTRACT_STATIC | PATCHBLOB_CONTRACT_DENSE | PATCHBLOB_CONTRACT_FP16 |
                   PATCHBLOB_CONTRACT_CMX | PATCHBLOB_CONTRACT_DISJOINT_OUTPUT;
  size_t index;
  npunlock_status status;

  carrier = read_file(FIXTURE_ROOT "abs-add-1x16-tile1.blob");
  two_operation_carrier = read_file(FIXTURE_ROOT "abs-abs-1x32.blob");
  broadcast_carrier = read_file(FIXTURE_ROOT "maximum-broadcast-1x32-f16.blob");
  convert_f32_to_f16 = read_file(FIXTURE_ROOT "convert-f32-to-f16-1x16.blob");
  convert_f16_to_f32 = read_file(FIXTURE_ROOT "convert-f16-to-f32-1x16.blob");
  i32_carrier = read_file(FIXTURE_ROOT "abs-i32-1x32.blob");
  i8_carrier = read_file(FIXTURE_ROOT "abs-i8-1x32.blob");
  elf = read_file(FIXTURE_ROOT "add1-fp16.elf");
  expected = read_file(FIXTURE_ROOT "add1-shared-1x16.blob");
  CHECK(carrier.data != NULL && two_operation_carrier.data != NULL &&
        broadcast_carrier.data != NULL && convert_f32_to_f16.data != NULL &&
        convert_f16_to_f32.data != NULL && i32_carrier.data != NULL && i8_carrier.data != NULL &&
        elf.data != NULL && expected.data != NULL);
  unsupported_header.data = (uint8_t *)malloc(carrier.size);
  CHECK(unsupported_header.data != NULL);
  unsupported_header.size = carrier.size;
  memcpy(unsupported_header.data, carrier.data, carrier.size);
  unsupported_header.data[0x10] = 2u;
  status = patchblob_discover_targets(
      (npunlock_view){unsupported_header.data, unsupported_header.size}, &discovery);
  CHECK(status == NPUNLOCK_STATUS_UNSUPPORTED);
  CHECK(discovery.diagnostic.json.data != NULL);
  CHECK(strstr((const char *)discovery.diagnostic.json.data, "ELF identity") != NULL);
  patchblob_discovery_result_release(&discovery);
  unsupported_carrier.data = (uint8_t *)malloc(carrier.size);
  CHECK(unsupported_carrier.data != NULL);
  unsupported_carrier.size = carrier.size;
  memcpy(unsupported_carrier.data, carrier.data, carrier.size);
  {
    static const uint8_t version_text[] = "compiler=8.3;npu=3720";
    uint8_t *version = find_bytes(unsupported_carrier.data, unsupported_carrier.size, version_text,
                                  sizeof(version_text) - 1u);
    CHECK(version != NULL);
    version[11] = '4';
  }
  status = patchblob_discover_targets(
      (npunlock_view){unsupported_carrier.data, unsupported_carrier.size}, &discovery);
  CHECK(status == NPUNLOCK_STATUS_UNSUPPORTED);
  CHECK(discovery.diagnostic.json.data != NULL);
  CHECK(strstr((const char *)discovery.diagnostic.json.data, "supported 8.3 contract") != NULL);
  patchblob_discovery_result_release(&discovery);
  unsupported_tiles.data = (uint8_t *)malloc(two_operation_carrier.size);
  CHECK(unsupported_tiles.data != NULL);
  unsupported_tiles.size = two_operation_carrier.size;
  memcpy(unsupported_tiles.data, two_operation_carrier.data, two_operation_carrier.size);
  {
    static const uint8_t tile_text[] = ";t=2;elf=";
    uint8_t *tiles = find_bytes(unsupported_tiles.data, unsupported_tiles.size, tile_text,
                                sizeof(tile_text) - 1u);
    CHECK(tiles != NULL);
    tiles[3] = '1';
  }
  status = patchblob_discover_targets(
      (npunlock_view){unsupported_tiles.data, unsupported_tiles.size}, &discovery);
  CHECK(status == NPUNLOCK_STATUS_UNSUPPORTED);
  CHECK(discovery.diagnostic.json.data != NULL);
  CHECK(strstr((const char *)discovery.diagnostic.json.data, "tile index exceeds") != NULL);
  patchblob_discovery_result_release(&discovery);
  unsupported_platform.data = (uint8_t *)malloc(carrier.size);
  CHECK(unsupported_platform.data != NULL);
  unsupported_platform.size = carrier.size;
  memcpy(unsupported_platform.data, carrier.data, carrier.size);
  {
    static const uint8_t platform_info[40] = {
        0x20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
        0,    0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0,
    };
    uint8_t *platform = find_bytes(unsupported_platform.data, unsupported_platform.size,
                                   platform_info, sizeof(platform_info));
    CHECK(platform != NULL);
    platform[32] = 4u;
  }
  status = patchblob_discover_targets(
      (npunlock_view){unsupported_platform.data, unsupported_platform.size}, &discovery);
  CHECK(status == NPUNLOCK_STATUS_UNSUPPORTED);
  CHECK(discovery.diagnostic.json.data != NULL);
  CHECK(strstr((const char *)discovery.diagnostic.json.data, "supported NPU3720 contract") != NULL);
  patchblob_discovery_result_release(&discovery);
  malformed_sections.data = (uint8_t *)malloc(carrier.size);
  CHECK(malformed_sections.data != NULL);
  malformed_sections.size = carrier.size;
  memcpy(malformed_sections.data, carrier.data, carrier.size);
  {
    uint8_t *params = find_section_header(malformed_sections.data, malformed_sections.size,
                                          ".text.KernelParams", UINT32_MAX);
    CHECK(params != NULL);
    write_u64(params + 0x18u, read_u64(params + 0x18u) + 1u);
  }
  status = patchblob_discover_targets(
      (npunlock_view){malformed_sections.data, malformed_sections.size}, &discovery);
  CHECK(status == NPUNLOCK_STATUS_MALFORMED_INPUT);
  CHECK(discovery.diagnostic.json.data != NULL);
  CHECK(strstr((const char *)discovery.diagnostic.json.data, "alignment") != NULL);
  patchblob_discovery_result_release(&discovery);
  memcpy(malformed_sections.data, carrier.data, carrier.size);
  {
    uint8_t *params = find_section_header(malformed_sections.data, malformed_sections.size,
                                          ".text.KernelParams", UINT32_MAX);
    uint8_t *ranges = find_section_header(malformed_sections.data, malformed_sections.size,
                                          ".text.ActKernelRanges", UINT32_MAX);
    CHECK(params != NULL && ranges != NULL);
    write_u64(params + 0x18u, read_u64(ranges + 0x18u));
  }
  status = patchblob_discover_targets(
      (npunlock_view){malformed_sections.data, malformed_sections.size}, &discovery);
  CHECK(status == NPUNLOCK_STATUS_MALFORMED_INPUT);
  CHECK(discovery.diagnostic.json.data != NULL);
  CHECK(strstr((const char *)discovery.diagnostic.json.data, "overlap") != NULL);
  patchblob_discovery_result_release(&discovery);
  memcpy(malformed_sections.data, carrier.data, carrier.size);
  {
    uint8_t *relocations = find_section_header(malformed_sections.data, malformed_sections.size,
                                               ".rlt.text.KernelParams", 0xff20u);
    CHECK(relocations != NULL);
    write_u32(relocations + 0x28u, 0xff21u);
  }
  status = patchblob_discover_targets(
      (npunlock_view){malformed_sections.data, malformed_sections.size}, &discovery);
  CHECK(status == NPUNLOCK_STATUS_UNSUPPORTED);
  CHECK(discovery.diagnostic.json.data != NULL);
  CHECK(strstr((const char *)discovery.diagnostic.json.data, "VPU runtime symbol") != NULL);
  patchblob_discovery_result_release(&discovery);
  memcpy(malformed_sections.data, carrier.data, carrier.size);
  {
    uint8_t *relocations = find_section_header(malformed_sections.data, malformed_sections.size,
                                               ".rlt.text.KernelParams", 0xff20u);
    size_t relocation_offset;
    uint8_t *entry;
    CHECK(relocations != NULL);
    CHECK(read_u64(relocations + 0x18u) <= SIZE_MAX);
    relocation_offset = (size_t)read_u64(relocations + 0x18u);
    CHECK(relocation_offset <= malformed_sections.size - 24u);
    entry = malformed_sections.data + relocation_offset;
    write_u64(entry + 8u, (1ull << 32) | read_u32(entry + 8u));
  }
  status = patchblob_discover_targets(
      (npunlock_view){malformed_sections.data, malformed_sections.size}, &discovery);
  CHECK(status == NPUNLOCK_STATUS_UNSUPPORTED);
  CHECK(discovery.diagnostic.json.data != NULL);
  CHECK(strstr((const char *)discovery.diagnostic.json.data,
               "discoverable input/output contract") != NULL);
  patchblob_discovery_result_release(&discovery);
  memcpy(malformed_sections.data, carrier.data, carrier.size);
  {
    uint8_t *symbols = find_section_header(malformed_sections.data, malformed_sections.size,
                                           ".symtab.tasks", UINT32_MAX);
    size_t section_table = (size_t)read_u64(malformed_sections.data + 0x28u);
    uint32_t symbol_section_index;
    uint8_t *relocations;
    size_t relocation_offset;
    size_t symbol_offset;
    uint32_t symbol_index;
    CHECK(symbols != NULL);
    CHECK(symbols >= malformed_sections.data + section_table);
    symbol_section_index =
        (uint32_t)((size_t)(symbols - malformed_sections.data - section_table) / 64u);
    relocations = find_section_header(malformed_sections.data, malformed_sections.size,
                                      ".rlt.text.KernelParams", symbol_section_index);
    CHECK(relocations != NULL);
    CHECK(read_u64(relocations + 0x18u) <= SIZE_MAX);
    relocation_offset = (size_t)read_u64(relocations + 0x18u);
    CHECK(relocation_offset <= malformed_sections.size - 24u);
    symbol_index = (uint32_t)(read_u64(malformed_sections.data + relocation_offset + 8u) >> 32);
    CHECK(read_u64(symbols + 0x18u) <= SIZE_MAX);
    symbol_offset = (size_t)read_u64(symbols + 0x18u) + (size_t)symbol_index * 24u;
    CHECK(symbol_offset <= malformed_sections.size - 24u);
    malformed_sections.data[symbol_offset + 4u] = 1u;
  }
  status = patchblob_discover_targets(
      (npunlock_view){malformed_sections.data, malformed_sections.size}, &discovery);
  CHECK(status == NPUNLOCK_STATUS_UNSUPPORTED);
  CHECK(discovery.diagnostic.json.data != NULL);
  CHECK(strstr((const char *)discovery.diagnostic.json.data, "unsupported source symbol") != NULL);
  patchblob_discovery_result_release(&discovery);
  memcpy(malformed_sections.data, carrier.data, carrier.size);
  {
    uint8_t *params = find_section_header(malformed_sections.data, malformed_sections.size,
                                          ".text.KernelParams", UINT32_MAX);
    uint8_t *relocations = find_section_header(malformed_sections.data, malformed_sections.size,
                                               ".rlt.text.KernelParams", 0xff20u);
    size_t relocation_offset;
    CHECK(params != NULL && relocations != NULL);
    CHECK(read_u64(relocations + 0x18u) <= SIZE_MAX);
    relocation_offset = (size_t)read_u64(relocations + 0x18u);
    CHECK(relocation_offset <= malformed_sections.size - 24u);
    write_u64(malformed_sections.data + relocation_offset, read_u64(params + 0x20u));
  }
  status = patchblob_discover_targets(
      (npunlock_view){malformed_sections.data, malformed_sections.size}, &discovery);
  CHECK(status == NPUNLOCK_STATUS_MALFORMED_INPUT);
  CHECK(discovery.diagnostic.json.data != NULL);
  CHECK(strstr((const char *)discovery.diagnostic.json.data, "outside or misaligned") != NULL);
  patchblob_discovery_result_release(&discovery);
  memcpy(malformed_sections.data, carrier.data, carrier.size);
  {
    uint8_t *relocations = find_section_header(malformed_sections.data, malformed_sections.size,
                                               ".rlt.text.KernelParams", 0xff20u);
    size_t relocation_offset;
    uint64_t info;
    CHECK(relocations != NULL);
    CHECK(read_u64(relocations + 0x18u) <= SIZE_MAX);
    relocation_offset = (size_t)read_u64(relocations + 0x18u);
    CHECK(relocation_offset <= malformed_sections.size - 24u);
    info = read_u64(malformed_sections.data + relocation_offset + 8u);
    write_u64(malformed_sections.data + relocation_offset + 8u,
              (info & 0xffffffff00000000ull) | 6u);
  }
  status = patchblob_discover_targets(
      (npunlock_view){malformed_sections.data, malformed_sections.size}, &discovery);
  CHECK(status == NPUNLOCK_STATUS_UNSUPPORTED);
  CHECK(discovery.diagnostic.json.data != NULL);
  CHECK(strstr((const char *)discovery.diagnostic.json.data, "unsupported relocation type") !=
        NULL);
  patchblob_discovery_result_release(&discovery);
  status = patchblob_discover_targets((npunlock_view){carrier.data, carrier.size}, &discovery);
  if (status != NPUNLOCK_STATUS_OK && discovery.diagnostic.json.data != NULL) {
    fprintf(stderr, "%s\n", (const char *)discovery.diagnostic.json.data);
  }
  CHECK(status == NPUNLOCK_STATUS_OK);
  CHECK(discovery.group_count == 1);
  CHECK(discovery.target_count == 2);
  for (index = 0; index < discovery.target_count; ++index) {
    CHECK(discovery.targets[index].group_index == 0);
    CHECK(discovery.targets[index].target.invocation_index == index);
    CHECK(discovery.targets[index].target.range_index == index);
    CHECK(discovery.targets[index].target.expected_input_count == 1);
    CHECK(discovery.targets[index].target.expected_element_count == 8);
    CHECK(discovery.targets[index].target.expected_span_bytes == 16);
  }
  patchblob_discovery_result_release(&discovery);
  status =
      patchblob_discover_targets((npunlock_view){i32_carrier.data, i32_carrier.size}, &discovery);
  CHECK(status == NPUNLOCK_STATUS_UNSUPPORTED);
  patchblob_discovery_result_release(&discovery);
  status = patchblob_discover_targets_v2((npunlock_view){i32_carrier.data, i32_carrier.size},
                                         &discovery_v2);
  if (status != NPUNLOCK_STATUS_OK && discovery_v2.diagnostic.json.data != NULL) {
    fprintf(stderr, "%s\n", (const char *)discovery_v2.diagnostic.json.data);
  }
  CHECK(status == NPUNLOCK_STATUS_OK);
  CHECK(discovery_v2.group_count == 1);
  CHECK(discovery_v2.target_count == 4);
  for (index = 0; index < discovery_v2.target_count; ++index) {
    const patchblob_target_v2 *target = &discovery_v2.targets[index].target;
    CHECK(discovery_v2.targets[index].group_index == 0);
    CHECK(target->invocation_index == index);
    CHECK(target->range_index == index);
    CHECK(target->target_flags == 0);
    CHECK(target->tensor_count == 2);
    CHECK(target->tensors[0].role == PATCHBLOB_TENSOR_ROLE_INPUT_V2);
    CHECK(target->tensors[0].precision == PATCHBLOB_TENSOR_PRECISION_I32_V2);
    CHECK(target->tensors[0].element_count == 16);
    CHECK(target->tensors[0].span_bytes == 64);
    CHECK(target->tensors[1].role == PATCHBLOB_TENSOR_ROLE_OUTPUT_V2);
    CHECK(target->tensors[1].precision == PATCHBLOB_TENSOR_PRECISION_I32_V2);
    CHECK(target->tensors[1].element_count == 16);
    CHECK(target->tensors[1].span_bytes == 64);
  }
  patchblob_discovery_result_v2_release(&discovery_v2);
  status =
      patchblob_discover_targets((npunlock_view){i8_carrier.data, i8_carrier.size}, &discovery);
  CHECK(status == NPUNLOCK_STATUS_UNSUPPORTED);
  patchblob_discovery_result_release(&discovery);
  status = patchblob_discover_targets_v2((npunlock_view){i8_carrier.data, i8_carrier.size},
                                         &discovery_v2);
  if (status != NPUNLOCK_STATUS_OK && discovery_v2.diagnostic.json.data != NULL) {
    fprintf(stderr, "%s\n", (const char *)discovery_v2.diagnostic.json.data);
  }
  CHECK(status == NPUNLOCK_STATUS_OK);
  CHECK(discovery_v2.group_count == 1);
  CHECK(discovery_v2.target_count == 4);
  for (index = 0; index < discovery_v2.target_count; ++index) {
    const patchblob_target_v2 *target = &discovery_v2.targets[index].target;
    CHECK(discovery_v2.targets[index].group_index == 0);
    CHECK(target->invocation_index == index);
    CHECK(target->range_index == index);
    CHECK(target->target_flags == 0);
    CHECK(target->tensor_count == 2);
    CHECK(target->tensors[0].role == PATCHBLOB_TENSOR_ROLE_INPUT_V2);
    CHECK(target->tensors[0].precision == PATCHBLOB_TENSOR_PRECISION_I8_V2);
    CHECK(target->tensors[0].element_count == 16);
    CHECK(target->tensors[0].span_bytes == 16);
    CHECK(target->tensors[1].role == PATCHBLOB_TENSOR_ROLE_OUTPUT_V2);
    CHECK(target->tensors[1].precision == PATCHBLOB_TENSOR_PRECISION_I8_V2);
    CHECK(target->tensors[1].element_count == 16);
    CHECK(target->tensors[1].span_bytes == 16);
  }
  patchblob_discovery_result_v2_release(&discovery_v2);
  status = patchblob_discover_targets(
      (npunlock_view){convert_f32_to_f16.data, convert_f32_to_f16.size}, &discovery);
  CHECK(status == NPUNLOCK_STATUS_UNSUPPORTED);
  patchblob_discovery_result_release(&discovery);
  status = patchblob_discover_targets_v2(
      (npunlock_view){convert_f32_to_f16.data, convert_f32_to_f16.size}, &discovery_v2);
  if (status != NPUNLOCK_STATUS_OK && discovery_v2.diagnostic.json.data != NULL) {
    fprintf(stderr, "%s\n", (const char *)discovery_v2.diagnostic.json.data);
  }
  CHECK(status == NPUNLOCK_STATUS_OK);
  CHECK(discovery_v2.group_count == 2);
  CHECK(discovery_v2.target_count == 6);
  for (index = 0; index < 2; ++index) {
    const patchblob_target_v2 *target = &discovery_v2.targets[index].target;
    CHECK(discovery_v2.targets[index].group_index == 0);
    CHECK(target->target_flags == PATCHBLOB_TARGET_UNARY_CONVERSION_V2);
    CHECK(target->tensor_count == 2);
    CHECK(target->tensors[0].role == PATCHBLOB_TENSOR_ROLE_INPUT_V2);
    CHECK(target->tensors[0].precision == PATCHBLOB_TENSOR_PRECISION_FP32_V2);
    CHECK(target->tensors[0].element_count == 16);
    CHECK(target->tensors[0].span_bytes == 64);
    CHECK(target->tensors[1].role == PATCHBLOB_TENSOR_ROLE_OUTPUT_V2);
    CHECK(target->tensors[1].precision == PATCHBLOB_TENSOR_PRECISION_FP16_V2);
    CHECK(target->tensors[1].element_count == 16);
    CHECK(target->tensors[1].span_bytes == 32);
    conversion_targets[index] = *target;
  }
  options.struct_size = sizeof(options);
  options.image_alignment = 0x400;
  options.tail_padding = 0x80;
  status = patchblob_patch_v2(&options,
                              (npunlock_view){convert_f32_to_f16.data, convert_f32_to_f16.size},
                              (npunlock_view){elf.data, elf.size}, conversion_targets, 1, &result);
  CHECK(status == NPUNLOCK_STATUS_UNSUPPORTED);
  CHECK(strstr((const char *)result.diagnostic.json.data, "complete ACT replica group") != NULL);
  patchblob_result_release(&result);
  status = patchblob_patch_v2(&options,
                              (npunlock_view){convert_f32_to_f16.data, convert_f32_to_f16.size},
                              (npunlock_view){elf.data, elf.size}, conversion_targets, 2, &result);
  CHECK(status == NPUNLOCK_STATUS_OK);
  CHECK(result.graph_blob.size > convert_f32_to_f16.size);
  CHECK(strstr((const char *)result.report_json.data, "npunlock.patchblob.v2") != NULL);
  CHECK(strstr((const char *)result.report_json.data, "\"contract_version\":2") != NULL);
  CHECK(strstr((const char *)result.report_json.data, "\"precision\":\"fp32\"") != NULL);
  CHECK(strstr((const char *)result.report_json.data, "\"precision\":\"fp16\"") != NULL);
  patchblob_result_release(&result);
  patchblob_discovery_result_v2_release(&discovery_v2);
  status = patchblob_discover_targets_v2(
      (npunlock_view){convert_f16_to_f32.data, convert_f16_to_f32.size}, &discovery_v2);
  CHECK(status == NPUNLOCK_STATUS_OK);
  CHECK(discovery_v2.group_count == 2);
  CHECK(discovery_v2.target_count == 6);
  for (index = 4; index < 6; ++index) {
    const patchblob_target_v2 *target = &discovery_v2.targets[index].target;
    CHECK(discovery_v2.targets[index].group_index == 1);
    CHECK(target->target_flags == PATCHBLOB_TARGET_UNARY_CONVERSION_V2);
    CHECK(target->tensors[0].precision == PATCHBLOB_TENSOR_PRECISION_FP16_V2);
    CHECK(target->tensors[0].span_bytes == 32);
    CHECK(target->tensors[1].precision == PATCHBLOB_TENSOR_PRECISION_FP32_V2);
    CHECK(target->tensors[1].span_bytes == 64);
  }
  patchblob_discovery_result_v2_release(&discovery_v2);
  status = patchblob_discover_targets(
      (npunlock_view){two_operation_carrier.data, two_operation_carrier.size}, &discovery);
  if (status != NPUNLOCK_STATUS_OK && discovery.diagnostic.json.data != NULL) {
    fprintf(stderr, "%s\n", (const char *)discovery.diagnostic.json.data);
  }
  CHECK(status == NPUNLOCK_STATUS_OK);
  CHECK(discovery.group_count == 2);
  CHECK(discovery.target_count == 8);
  for (index = 0; index < discovery.target_count; ++index) {
    CHECK(discovery.targets[index].group_index == index / 4u);
    CHECK(discovery.targets[index].target.invocation_index == index);
    CHECK(discovery.targets[index].target.range_index == index);
    CHECK(discovery.targets[index].target.expected_input_count == 1);
    CHECK(discovery.targets[index].target.expected_element_count == 16);
    CHECK(discovery.targets[index].target.expected_span_bytes == 32);
  }
  patchblob_discovery_result_release(&discovery);
  status = patchblob_discover_targets(
      (npunlock_view){broadcast_carrier.data, broadcast_carrier.size}, &discovery);
  if (status != NPUNLOCK_STATUS_OK && discovery.diagnostic.json.data != NULL) {
    fprintf(stderr, "%s\n", (const char *)discovery.diagnostic.json.data);
  }
  CHECK(status == NPUNLOCK_STATUS_OK);
  CHECK(discovery.group_count == 1);
  CHECK(discovery.target_count == 4);
  for (index = 0; index < discovery.target_count; ++index) {
    const patchblob_target *target = &discovery.targets[index].target;
    CHECK(discovery.targets[index].group_index == 0);
    CHECK(target->invocation_index == index);
    CHECK(target->range_index == index);
    CHECK(target->expected_input_count == 2);
    CHECK(target->expected_element_count == 8);
    CHECK(target->expected_span_bytes == 16);
    CHECK((target->required_contract_flags & PATCHBLOB_CONTRACT_INPUT_1_SCALAR) != 0);
    broadcast_targets[index] = *target;
  }
  patchblob_discovery_result_release(&discovery);
  for (index = 0; index < 2; ++index) {
    targets[index].struct_size = sizeof(targets[index]);
    targets[index].invocation_index = (uint32_t)index;
    targets[index].range_index = (uint32_t)index;
    targets[index].expected_input_count = 1;
    targets[index].expected_element_count = 8;
    targets[index].expected_span_bytes = 16;
    targets[index].required_contract_flags = flags;
  }
  status = patchblob_patch(&options, (npunlock_view){carrier.data, carrier.size},
                           (npunlock_view){elf.data, elf.size}, targets, 2, &result);
  if (status != NPUNLOCK_STATUS_OK && result.diagnostic.json.data != NULL) {
    fprintf(stderr, "%s\n", (const char *)result.diagnostic.json.data);
  }
  CHECK(status == NPUNLOCK_STATUS_OK);
  CHECK(result.graph_blob.size == expected.size);
  CHECK(memcmp(result.graph_blob.data, expected.data, expected.size) == 0);
  CHECK(sections_are_aligned(result.graph_blob.data, result.graph_blob.size));
  CHECK(result.report_json.data != NULL);
  CHECK(strstr((const char *)result.report_json.data, "npunlock.patchblob.v1") != NULL);
  CHECK(strstr((const char *)result.report_json.data,
               "\"graph_contract\":{\"compiler\":\"8.3\",\"npu\":\"3720\",\"tiles\":1") != NULL);
  CHECK(strstr((const char *)result.report_json.data, "\"invocation_index\":1") != NULL);
  patchblob_result_release(&result);
  status =
      patchblob_patch(&options, (npunlock_view){broadcast_carrier.data, broadcast_carrier.size},
                      (npunlock_view){elf.data, elf.size}, broadcast_targets, 4, &result);
  if (status != NPUNLOCK_STATUS_OK && result.diagnostic.json.data != NULL) {
    fprintf(stderr, "%s\n", (const char *)result.diagnostic.json.data);
  }
  CHECK(status == NPUNLOCK_STATUS_OK);
  CHECK(result.graph_blob.size > broadcast_carrier.size);
  CHECK(strstr((const char *)result.report_json.data, "\"input_1_scalar\"") != NULL);
  patchblob_result_release(&result);
  broadcast_targets[0].required_contract_flags &= ~PATCHBLOB_CONTRACT_INPUT_1_SCALAR;
  status =
      patchblob_patch(&options, (npunlock_view){broadcast_carrier.data, broadcast_carrier.size},
                      (npunlock_view){elf.data, elf.size}, broadcast_targets, 4, &result);
  CHECK(status == NPUNLOCK_STATUS_UNSUPPORTED);
  CHECK(result.diagnostic.json.data != NULL);
  CHECK(strstr((const char *)result.diagnostic.json.data, "expected element contract") != NULL);
  patchblob_result_release(&result);
  free(expected.data);
  free(elf.data);
  free(unsupported_header.data);
  free(unsupported_carrier.data);
  free(unsupported_tiles.data);
  free(unsupported_platform.data);
  free(malformed_sections.data);
  free(broadcast_carrier.data);
  free(convert_f16_to_f32.data);
  free(convert_f32_to_f16.data);
  free(i32_carrier.data);
  free(i8_carrier.data);
  free(two_operation_carrier.data);
  free(carrier.data);
  return 0;
}
