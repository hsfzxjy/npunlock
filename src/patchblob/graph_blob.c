#include "graph_blob.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include "internal.h"

#define ELF64_HEADER_SIZE 64u
#define ELF64_SECTION_SIZE 64u
#define ELF64_SYMBOL_SIZE 24u
#define ELF64_RELA_SIZE 24u
#define ACT_INVOCATION_SIZE 0x40u
#define ACT_RANGE_SIZE 0x18u
#define MEMREF_SIZE 0x28u
#define GRAPH_CODE_ADDRESS 0x1d000000u
#define SHT_PROGBITS 1u
#define SHT_SYMTAB 2u
#define SHT_RELA 4u
#define SHT_NOBITS 8u
#define R_VPU_64 4u

typedef struct graph_section {
  uint32_t type;
  uint64_t flags;
  uint64_t address;
  size_t offset;
  size_t size;
  uint32_t link;
  uint32_t info;
  size_t alignment;
  size_t entry_size;
  size_t header_offset;
  const char *name;
} graph_section;

typedef struct graph_layout {
  npunlock_view blob;
  size_t section_table_offset;
  uint16_t section_count;
  uint16_t section_name_index;
  graph_section *sections;
  size_t kernel_text_index;
  size_t params_index;
  size_t invocations_index;
  size_t ranges_index;
  size_t range_relocations_index;
} graph_layout;

typedef struct graph_relocation {
  size_t file_offset;
  uint64_t target_offset;
  uint32_t type;
  uint32_t symbol;
  int64_t addend;
  bool special_source;
  size_t source_section_index;
} graph_relocation;

typedef struct memref_contract {
  uint64_t element_count;
  uint64_t span_bytes;
  uint32_t dtype_kind;
  uint32_t data_symbol;
  int64_t data_addend;
} memref_contract;

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

static int32_t read_i32(const uint8_t *bytes) { return (int32_t)read_u32(bytes); }

static int64_t read_i64(const uint8_t *bytes) { return (int64_t)read_u64(bytes); }

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

static bool range_in_bounds(size_t offset, size_t extent, size_t size) {
  size_t end;
  return npunlock_checked_add_size(offset, extent, &end) && end <= size;
}

static bool u64_to_size(uint64_t value, size_t *result) {
  if (value > (uint64_t)SIZE_MAX) {
    return false;
  }
  *result = (size_t)value;
  return true;
}

static bool align_up(size_t value, size_t alignment, size_t *result) {
  size_t adjusted;
  if (alignment == 0 || (alignment & (alignment - 1)) != 0 ||
      !npunlock_checked_add_size(value, alignment - 1, &adjusted)) {
    return false;
  }
  *result = adjusted & ~(alignment - 1);
  return true;
}

static npunlock_status malformed(npunlock_diagnostic *diagnostic, const char *message) {
  return npunlock_set_diagnostic(diagnostic, NPUNLOCK_STATUS_MALFORMED_INPUT,
                                 "patchblob.parse_graph", message);
}

static npunlock_status unsupported(npunlock_diagnostic *diagnostic, const char *message) {
  return npunlock_set_diagnostic(diagnostic, NPUNLOCK_STATUS_UNSUPPORTED,
                                 "patchblob.validate_contract", message);
}

static void graph_layout_release(graph_layout *layout) {
  free(layout->sections);
  memset(layout, 0, sizeof(*layout));
}

static npunlock_status parse_graph(npunlock_view blob, graph_layout *layout,
                                   npunlock_diagnostic *diagnostic) {
  const uint8_t *bytes = blob.data;
  uint64_t section_table_u64;
  size_t table_size;
  size_t index;
  size_t string_offset;
  size_t string_size;

  memset(layout, 0, sizeof(*layout));
  layout->kernel_text_index = SIZE_MAX;
  layout->params_index = SIZE_MAX;
  layout->invocations_index = SIZE_MAX;
  layout->ranges_index = SIZE_MAX;
  layout->range_relocations_index = SIZE_MAX;
  if (blob.size < ELF64_HEADER_SIZE ||
      memcmp(bytes,
             "\x7f"
             "ELF",
             4) != 0 ||
      bytes[4] != 2 || bytes[5] != 1) {
    return malformed(diagnostic, "graph blob is not a supported ELF64 little-endian image");
  }
  if (read_u16(bytes + 0x38) != 0 || read_u16(bytes + 0x3a) != ELF64_SECTION_SIZE) {
    return unsupported(diagnostic,
                       "graph blobs with program headers or nonstandard sections are unsupported");
  }
  section_table_u64 = read_u64(bytes + 0x28);
  layout->section_count = read_u16(bytes + 0x3c);
  layout->section_name_index = read_u16(bytes + 0x3e);
  if (!u64_to_size(section_table_u64, &layout->section_table_offset) ||
      layout->section_count == 0 || layout->section_name_index >= layout->section_count ||
      !npunlock_checked_mul_size(layout->section_count, ELF64_SECTION_SIZE, &table_size) ||
      !range_in_bounds(layout->section_table_offset, table_size, blob.size)) {
    return malformed(diagnostic, "graph section table is outside the input blob");
  }
  layout->sections = (graph_section *)calloc(layout->section_count, sizeof(*layout->sections));
  if (layout->sections == NULL) {
    return npunlock_set_diagnostic(diagnostic, NPUNLOCK_STATUS_OUT_OF_MEMORY,
                                   "patchblob.parse_graph", "could not allocate section metadata");
  }
  layout->blob = blob;
  for (index = 0; index < layout->section_count; ++index) {
    graph_section *section = &layout->sections[index];
    const uint8_t *header = bytes + layout->section_table_offset + index * ELF64_SECTION_SIZE;
    section->header_offset = layout->section_table_offset + index * ELF64_SECTION_SIZE;
    section->type = read_u32(header + 4);
    section->flags = read_u64(header + 8);
    section->address = read_u64(header + 0x10);
    section->link = read_u32(header + 0x28);
    section->info = read_u32(header + 0x2c);
    if (!u64_to_size(read_u64(header + 0x18), &section->offset) ||
        !u64_to_size(read_u64(header + 0x20), &section->size) ||
        !u64_to_size(read_u64(header + 0x30), &section->alignment) ||
        !u64_to_size(read_u64(header + 0x38), &section->entry_size) ||
        (section->type != SHT_NOBITS &&
         !range_in_bounds(section->offset, section->size, blob.size))) {
      graph_layout_release(layout);
      return malformed(diagnostic, "graph section extent is outside the input blob");
    }
  }
  string_offset = layout->sections[layout->section_name_index].offset;
  string_size = layout->sections[layout->section_name_index].size;
  if (layout->sections[layout->section_name_index].type != 3u || string_size == 0) {
    graph_layout_release(layout);
    return malformed(diagnostic, "graph section-name string table is malformed");
  }
  for (index = 0; index < layout->section_count; ++index) {
    graph_section *section = &layout->sections[index];
    const uint8_t *header = bytes + section->header_offset;
    uint32_t name_offset = read_u32(header);
    const void *terminator;
    if (name_offset >= string_size) {
      graph_layout_release(layout);
      return malformed(diagnostic, "graph section name is outside the string table");
    }
    section->name = (const char *)(bytes + string_offset + name_offset);
    terminator = memchr(section->name, 0, string_size - name_offset);
    if (terminator == NULL) {
      graph_layout_release(layout);
      return malformed(diagnostic, "graph section name is not terminated");
    }
#define MATCH_REQUIRED(field, literal)                                                             \
  if (strcmp(section->name, literal) == 0) {                                                       \
    if (layout->field != SIZE_MAX) {                                                               \
      graph_layout_release(layout);                                                                \
      return malformed(diagnostic, "graph contains a duplicate required section");                 \
    }                                                                                              \
    layout->field = index;                                                                         \
  }
    MATCH_REQUIRED(kernel_text_index, ".text.KernelText")
    MATCH_REQUIRED(params_index, ".text.KernelParams")
    MATCH_REQUIRED(invocations_index, ".text.ActKernelInvocations")
    MATCH_REQUIRED(ranges_index, ".text.ActKernelRanges")
    MATCH_REQUIRED(range_relocations_index, ".rlt.text.ActKernelRanges")
#undef MATCH_REQUIRED
  }
  if (layout->kernel_text_index == SIZE_MAX || layout->params_index == SIZE_MAX ||
      layout->invocations_index == SIZE_MAX || layout->ranges_index == SIZE_MAX ||
      layout->range_relocations_index == SIZE_MAX) {
    graph_layout_release(layout);
    return unsupported(diagnostic, "graph does not contain the required ACT carrier sections");
  }
  if (layout->sections[layout->kernel_text_index].type != SHT_PROGBITS ||
      layout->sections[layout->params_index].type != SHT_PROGBITS ||
      layout->sections[layout->invocations_index].type != SHT_PROGBITS ||
      layout->sections[layout->ranges_index].type != SHT_PROGBITS ||
      layout->sections[layout->range_relocations_index].type != SHT_RELA) {
    graph_layout_release(layout);
    return unsupported(diagnostic, "required ACT carrier sections have unsupported types");
  }
  if (layout->sections[layout->invocations_index].size % ACT_INVOCATION_SIZE != 0 ||
      layout->sections[layout->ranges_index].size % ACT_RANGE_SIZE != 0 ||
      layout->sections[layout->range_relocations_index].entry_size != ELF64_RELA_SIZE ||
      layout->sections[layout->range_relocations_index].size % ELF64_RELA_SIZE != 0) {
    graph_layout_release(layout);
    return malformed(diagnostic, "ACT carrier section sizes are malformed");
  }
  return NPUNLOCK_STATUS_OK;
}

static npunlock_status resolve_relocation(const graph_layout *layout, size_t relocation_index,
                                          size_t entry_index, graph_relocation *relocation,
                                          npunlock_diagnostic *diagnostic) {
  const graph_section *relocations;
  const graph_section *symbols;
  const uint8_t *entry;
  uint64_t info;
  size_t symbol_count;
  const uint8_t *symbol;
  uint16_t source_index;

  if (relocation_index >= layout->section_count) {
    return malformed(diagnostic, "relocation section index is invalid");
  }
  relocations = &layout->sections[relocation_index];
  if (relocations->type != SHT_RELA || relocations->entry_size != ELF64_RELA_SIZE ||
      entry_index >= relocations->size / ELF64_RELA_SIZE) {
    return malformed(diagnostic, "relocation entry is malformed");
  }
  relocation->file_offset = relocations->offset + entry_index * ELF64_RELA_SIZE;
  entry = layout->blob.data + relocation->file_offset;
  relocation->target_offset = read_u64(entry);
  info = read_u64(entry + 8);
  relocation->symbol = (uint32_t)(info >> 32);
  relocation->type = (uint32_t)info;
  relocation->addend = read_i64(entry + 16);
  relocation->special_source = false;
  relocation->source_section_index = SIZE_MAX;
  if (relocations->link >= layout->section_count) {
    relocation->special_source = true;
    return NPUNLOCK_STATUS_OK;
  }
  symbols = &layout->sections[relocations->link];
  if (symbols->type != SHT_SYMTAB || symbols->entry_size != ELF64_SYMBOL_SIZE ||
      symbols->size % ELF64_SYMBOL_SIZE != 0) {
    return malformed(diagnostic, "relocation symbol table is malformed");
  }
  symbol_count = symbols->size / ELF64_SYMBOL_SIZE;
  if (relocation->symbol >= symbol_count) {
    return malformed(diagnostic, "relocation symbol index is outside the symbol table");
  }
  symbol = layout->blob.data + symbols->offset + relocation->symbol * ELF64_SYMBOL_SIZE;
  source_index = read_u16(symbol + 6);
  if (read_u64(symbol + 8) != 0 || source_index >= layout->section_count) {
    return unsupported(diagnostic, "relocation uses an unsupported source symbol");
  }
  relocation->source_section_index = source_index;
  return NPUNLOCK_STATUS_OK;
}

static npunlock_status find_pointer_relocation(const graph_layout *layout, size_t target_index,
                                               uint64_t target_offset, size_t source_index,
                                               bool require_special, graph_relocation *found,
                                               npunlock_diagnostic *diagnostic) {
  size_t section_index;
  size_t match_count = 0;
  for (section_index = 0; section_index < layout->section_count; ++section_index) {
    const graph_section *section = &layout->sections[section_index];
    size_t entry_index;
    if (section->type != SHT_RELA || section->info != target_index ||
        section->entry_size != ELF64_RELA_SIZE || section->size % ELF64_RELA_SIZE != 0) {
      continue;
    }
    for (entry_index = 0; entry_index < section->size / ELF64_RELA_SIZE; ++entry_index) {
      graph_relocation candidate;
      npunlock_status status =
          resolve_relocation(layout, section_index, entry_index, &candidate, diagnostic);
      if (status != NPUNLOCK_STATUS_OK) {
        return status;
      }
      if (candidate.target_offset == target_offset && candidate.type == R_VPU_64 &&
          ((require_special && candidate.special_source) ||
           (!require_special && !candidate.special_source &&
            candidate.source_section_index == source_index))) {
        *found = candidate;
        ++match_count;
      }
    }
  }
  if (match_count != 1) {
    return unsupported(diagnostic, "required pointer relocation is missing or ambiguous");
  }
  return NPUNLOCK_STATUS_OK;
}

static npunlock_status invocation_parameter_bases(const graph_layout *layout, uint64_t **bases_out,
                                                  size_t *count_out,
                                                  npunlock_diagnostic *diagnostic) {
  const graph_section *invocations = &layout->sections[layout->invocations_index];
  const graph_section *params = &layout->sections[layout->params_index];
  size_t count = invocations->size / ACT_INVOCATION_SIZE;
  uint64_t *bases = (uint64_t *)calloc(count, sizeof(*bases));
  size_t index;
  if (bases == NULL && count != 0) {
    return npunlock_set_diagnostic(diagnostic, NPUNLOCK_STATUS_OUT_OF_MEMORY,
                                   "patchblob.validate_contract",
                                   "could not allocate invocation metadata");
  }
  for (index = 0; index < count; ++index) {
    graph_relocation relocation;
    npunlock_status status = find_pointer_relocation(
        layout, layout->invocations_index, (uint64_t)index * ACT_INVOCATION_SIZE + 4u,
        layout->params_index, false, &relocation, diagnostic);
    if (status != NPUNLOCK_STATUS_OK) {
      free(bases);
      return status;
    }
    if (relocation.addend < 0 || (uint64_t)relocation.addend >= params->size) {
      free(bases);
      return unsupported(diagnostic, "invocation parameter base is outside KernelParams");
    }
    bases[index] = (uint64_t)relocation.addend;
  }
  *bases_out = bases;
  *count_out = count;
  return NPUNLOCK_STATUS_OK;
}

enum npunlock_dtype_kind {
  NPUNLOCK_DTYPE_F32 = 1,
  NPUNLOCK_DTYPE_F16 = 2,
  NPUNLOCK_DTYPE_MAX,
};

struct npunlock_dtype_info {
  uint32_t stride_bits;
  uint32_t size_bytes;
};

static const struct npunlock_dtype_info dtype_table[NPUNLOCK_DTYPE_MAX] = {
    [NPUNLOCK_DTYPE_F32] = {.stride_bits = 32, .size_bytes = 4},
    [NPUNLOCK_DTYPE_F16] = {.stride_bits = 16, .size_bytes = 2},
};

static npunlock_status validate_memref(const graph_layout *layout, uint64_t record_offset,
                                       memref_contract *contract, npunlock_diagnostic *diagnostic) {
  const graph_section *params = &layout->sections[layout->params_index];
  const uint8_t *record;
  uint32_t rank;
  graph_relocation dimensions;
  graph_relocation strides;
  graph_relocation data;
  int32_t dimension_values[15];
  int64_t stride_values[15];
  bool used[15] = {false};
  uint64_t count = 1;
  uint64_t expected_stride;
  uint64_t span;
  size_t index;
  if (record_offset > SIZE_MAX ||
      !range_in_bounds((size_t)record_offset, MEMREF_SIZE, params->size)) {
    return unsupported(diagnostic, "MemRefData record is outside KernelParams");
  }
  record = layout->blob.data + params->offset + (size_t)record_offset;
  uint32_t type_kind = read_u32(record + 0x18);
  if (read_u32(record + 8) != 1u || type_kind >= NPUNLOCK_DTYPE_MAX ||
      dtype_table[type_kind].stride_bits == 0 || read_u32(record + 0x24) != 2u) {
    return unsupported(diagnostic, "selected invocation is not static FP16/FP32 CMX");
  }
  expected_stride = dtype_table[type_kind].stride_bits;
  rank = read_u32(record + 0x0c);
  if (rank == 0 || rank > 15) {
    return unsupported(diagnostic, "selected invocation uses an unsupported tensor rank");
  }
  if (find_pointer_relocation(layout, layout->params_index, record_offset + 0x10u,
                              layout->params_index, false, &dimensions,
                              diagnostic) != NPUNLOCK_STATUS_OK ||
      find_pointer_relocation(layout, layout->params_index, record_offset + 0x14u,
                              layout->params_index, false, &strides,
                              diagnostic) != NPUNLOCK_STATUS_OK ||
      find_pointer_relocation(layout, layout->params_index, record_offset, SIZE_MAX, true, &data,
                              diagnostic) != NPUNLOCK_STATUS_OK) {
    return diagnostic->json.data == NULL
               ? unsupported(diagnostic, "MemRefData relocation is invalid")
               : NPUNLOCK_STATUS_UNSUPPORTED;
  }
  if (dimensions.addend < 0 || strides.addend < 0 ||
      !range_in_bounds((size_t)dimensions.addend, (size_t)rank * 4u, params->size) ||
      !range_in_bounds((size_t)strides.addend, (size_t)rank * 8u, params->size)) {
    return unsupported(diagnostic, "MemRefData dimensions or strides are outside KernelParams");
  }
  for (index = 0; index < rank; ++index) {
    int32_t dimension =
        read_i32(layout->blob.data + params->offset + (size_t)dimensions.addend + index * 4u);
    int64_t stride =
        read_i64(layout->blob.data + params->offset + (size_t)strides.addend + index * 8u);
    if (dimension <= 0 || stride < 0 || count > UINT64_MAX / (uint64_t)(uint32_t)dimension) {
      return unsupported(diagnostic, "MemRefData dimensions or strides are unsupported");
    }
    dimension_values[index] = dimension;
    stride_values[index] = stride;
    count *= (uint64_t)(uint32_t)dimension;
  }
  for (;;) {
    size_t selected = SIZE_MAX;
    for (index = 0; index < rank; ++index) {
      if (!used[index] && dimension_values[index] > 1 &&
          (selected == SIZE_MAX || stride_values[index] < stride_values[selected])) {
        selected = index;
      }
    }
    if (selected == SIZE_MAX) {
      break;
    }
    if ((uint64_t)stride_values[selected] != expected_stride ||
        expected_stride > UINT64_MAX / (uint64_t)(uint32_t)dimension_values[selected]) {
      return unsupported(diagnostic, "selected invocation uses a non-dense tensor layout");
    }
    expected_stride *= (uint64_t)(uint32_t)dimension_values[selected];
    used[selected] = true;
  }
  if (count > UINT64_MAX / dtype_table[type_kind].size_bytes) {
    return unsupported(diagnostic, "tensor byte span overflows");
  }
  span = count * dtype_table[type_kind].size_bytes;
  if (data.addend < 0) {
    return unsupported(diagnostic, "tensor data address is negative");
  }
  contract->element_count = count;
  contract->span_bytes = span;
  contract->dtype_kind = type_kind;
  contract->data_symbol = data.symbol;
  contract->data_addend = data.addend;
  return NPUNLOCK_STATUS_OK;
}

static bool spans_overlap(uint64_t left, uint64_t left_size, uint64_t right, uint64_t right_size) {
  uint64_t left_end;
  uint64_t right_end;
  if (left > UINT64_MAX - left_size || right > UINT64_MAX - right_size) {
    return true;
  }
  left_end = left + left_size;
  right_end = right + right_size;
  return left < right_end && right < left_end;
}

static uint64_t invocation_parameter_limit(const uint64_t *parameter_bases, size_t invocation_count,
                                           uint64_t base, size_t parameter_size) {
  uint64_t limit = parameter_size;
  size_t index;
  for (index = 0; index < invocation_count; ++index) {
    if (parameter_bases[index] > base && parameter_bases[index] < limit) {
      limit = parameter_bases[index];
    }
  }
  return limit;
}

static npunlock_status count_data_pointer_relocations(const graph_layout *layout, uint64_t base,
                                                      uint64_t limit, size_t *count,
                                                      npunlock_diagnostic *diagnostic) {
  size_t special_count = 0;
  size_t section_index;
  for (section_index = 0; section_index < layout->section_count; ++section_index) {
    const graph_section *section = &layout->sections[section_index];
    size_t relocation_index;
    if (section->type != SHT_RELA || section->info != layout->params_index ||
        section->entry_size != ELF64_RELA_SIZE || section->size % ELF64_RELA_SIZE != 0) {
      continue;
    }
    for (relocation_index = 0; relocation_index < section->size / ELF64_RELA_SIZE;
         ++relocation_index) {
      graph_relocation relocation = {0};
      npunlock_status status =
          resolve_relocation(layout, section_index, relocation_index, &relocation, diagnostic);
      if (status != NPUNLOCK_STATUS_OK) {
        return status;
      }
      if (relocation.special_source && relocation.type == R_VPU_64 &&
          relocation.target_offset >= base && relocation.target_offset < limit) {
        ++special_count;
      }
    }
  }
  *count = special_count;
  return NPUNLOCK_STATUS_OK;
}

static npunlock_status validate_target(const graph_layout *layout, const uint64_t *parameter_bases,
                                       size_t invocation_count, const patchblob_target *target,
                                       npunlock_patch_detail *detail,
                                       npunlock_diagnostic *diagnostic) {
  const graph_section *invocations = &layout->sections[layout->invocations_index];
  const graph_section *params = &layout->sections[layout->params_index];
  const graph_section *ranges = &layout->sections[layout->ranges_index];
  uint64_t base;
  uint64_t limit;
  uint64_t block_size;
  size_t invocation_offset;
  size_t range_offset;
  size_t expected_records;
  memref_contract inputs[8];
  memref_contract output;
  size_t index;
  uint32_t base_flags = PATCHBLOB_CONTRACT_STATIC | PATCHBLOB_CONTRACT_DENSE |
                        PATCHBLOB_CONTRACT_CMX | PATCHBLOB_CONTRACT_DISJOINT_OUTPUT;
  uint32_t precision_flags =
      target->required_contract_flags & (PATCHBLOB_CONTRACT_FP16 | PATCHBLOB_CONTRACT_FP32);
  uint32_t expected_dtype;
  if (target->invocation_index >= invocation_count ||
      target->range_index >= ranges->size / ACT_RANGE_SIZE || target->expected_input_count > 8u) {
    return unsupported(diagnostic, "selected invocation or range index is outside the ACT carrier");
  }
  if ((precision_flags != PATCHBLOB_CONTRACT_FP16 && precision_flags != PATCHBLOB_CONTRACT_FP32) ||
      target->required_contract_flags != (base_flags | precision_flags)) {
    return unsupported(diagnostic,
                       "MVP requires a complete static dense FP16 or FP32 CMX disjoint contract");
  }
  expected_dtype =
      precision_flags == PATCHBLOB_CONTRACT_FP16 ? NPUNLOCK_DTYPE_F16 : NPUNLOCK_DTYPE_F32;
  invocation_offset = invocations->offset + (size_t)target->invocation_index * ACT_INVOCATION_SIZE;
  if (read_u32(layout->blob.data + invocation_offset) != target->range_index) {
    return unsupported(diagnostic, "selected invocation does not reference the selected range");
  }
  base = parameter_bases[target->invocation_index];
  limit = invocation_parameter_limit(parameter_bases, invocation_count, base, params->size);
  block_size = limit - base;
  expected_records = (size_t)target->expected_input_count + 1u;
  if (block_size < expected_records * MEMREF_SIZE) {
    return unsupported(diagnostic, "selected invocation parameter block is too small");
  }
  for (index = 0; index < expected_records; ++index) {
    graph_relocation data;
    uint64_t slot = base + (uint64_t)index * MEMREF_SIZE;
    if (find_pointer_relocation(layout, layout->params_index, slot, SIZE_MAX, true, &data,
                                diagnostic) != NPUNLOCK_STATUS_OK) {
      return NPUNLOCK_STATUS_UNSUPPORTED;
    }
  }
  {
    size_t special_count;
    npunlock_status status =
        count_data_pointer_relocations(layout, base, limit, &special_count, diagnostic);
    if (status != NPUNLOCK_STATUS_OK) {
      return status;
    }
    if (special_count != expected_records) {
      return unsupported(
          diagnostic, "selected invocation parameter block has an unexpected data-pointer layout");
    }
  }
  for (index = 0; index < target->expected_input_count; ++index) {
    npunlock_status status =
        validate_memref(layout, base + (uint64_t)index * MEMREF_SIZE, &inputs[index], diagnostic);
    if (status != NPUNLOCK_STATUS_OK) {
      return status;
    }
    if (inputs[index].element_count != target->expected_element_count ||
        inputs[index].span_bytes != target->expected_span_bytes ||
        inputs[index].dtype_kind != expected_dtype) {
      return unsupported(diagnostic, "input tensor does not match the expected element contract");
    }
  }
  {
    npunlock_status status = validate_memref(
        layout, base + (uint64_t)target->expected_input_count * MEMREF_SIZE, &output, diagnostic);
    if (status != NPUNLOCK_STATUS_OK) {
      return status;
    }
  }
  if (output.element_count != target->expected_element_count ||
      output.span_bytes != target->expected_span_bytes || output.dtype_kind != expected_dtype) {
    return unsupported(diagnostic, "output tensor does not match the expected element contract");
  }
  for (index = 0; index < target->expected_input_count; ++index) {
    if (inputs[index].data_symbol != output.data_symbol) {
      return unsupported(diagnostic,
                         "selected invocation tensors do not share the observed address space");
    }
    if (spans_overlap((uint64_t)inputs[index].data_addend, inputs[index].span_bytes,
                      (uint64_t)output.data_addend, output.span_bytes)) {
      return unsupported(diagnostic, "selected invocation output overlaps an input tensor");
    }
  }
  range_offset = ranges->offset + (size_t)target->range_index * ACT_RANGE_SIZE;
  if (read_u32(layout->blob.data + range_offset + 4u) != GRAPH_CODE_ADDRESS ||
      read_u32(layout->blob.data + range_offset + 8u) != 0u) {
    return unsupported(diagnostic,
                       "selected range does not use the observed ACT code-address form");
  }
  memset(detail, 0, sizeof(*detail));
  detail->invocation_index = target->invocation_index;
  detail->range_index = target->range_index;
  detail->input_count = target->expected_input_count;
  detail->parameter_base = base;
  detail->element_count = target->expected_element_count;
  detail->span_bytes = target->expected_span_bytes;
  detail->extent_file_offset = range_offset + 0x0cu;
  detail->old_extent = read_u32(layout->blob.data + detail->extent_file_offset);
  return NPUNLOCK_STATUS_OK;
}

static npunlock_status find_range_relocation(const graph_layout *layout, uint32_t range_index,
                                             graph_relocation *found,
                                             npunlock_diagnostic *diagnostic) {
  const graph_section *section = &layout->sections[layout->range_relocations_index];
  size_t index;
  size_t matches = 0;
  if (section->info != layout->ranges_index || section->link >= layout->section_count) {
    return unsupported(diagnostic, "range relocation section has an unsupported link or target");
  }
  for (index = 0; index < section->size / ELF64_RELA_SIZE; ++index) {
    graph_relocation candidate;
    npunlock_status status =
        resolve_relocation(layout, layout->range_relocations_index, index, &candidate, diagnostic);
    if (status != NPUNLOCK_STATUS_OK) {
      return status;
    }
    if (candidate.target_offset == (uint64_t)range_index * ACT_RANGE_SIZE + 8u &&
        candidate.type == R_VPU_64 && !candidate.special_source &&
        candidate.source_section_index == layout->kernel_text_index) {
      *found = candidate;
      ++matches;
    }
  }
  if (matches != 1) {
    return unsupported(diagnostic, "selected range code relocation is missing or ambiguous");
  }
  return NPUNLOCK_STATUS_OK;
}

static bool invocation_group_identity_equal(const graph_layout *layout, size_t left_index,
                                            size_t right_index) {
  const graph_section *invocations = &layout->sections[layout->invocations_index];
  const uint8_t *left = layout->blob.data + invocations->offset + left_index * ACT_INVOCATION_SIZE;
  const uint8_t *right =
      layout->blob.data + invocations->offset + right_index * ACT_INVOCATION_SIZE;

  /*
   * Compiler 8.3 unary-chain observations identify one source operation by the
   * invariant, non-relocated slices below. Range, parameter, profiling, tile,
   * and per-invocation indices are deliberately excluded. Callers still check
   * the discovered group count against their symbolic graph, so an unfamiliar
   * layout fails closed instead of becoming a guessed source-node mapping.
   */
  return memcmp(left + 0x0cu, right + 0x0cu, 0x24u) == 0 &&
         memcmp(left + 0x3cu, right + 0x3cu, 4u) == 0;
}

npunlock_status npunlock_discover_graph_targets(npunlock_view graph_blob,
                                                patchblob_discovered_target **targets,
                                                size_t *target_count, size_t *group_count,
                                                npunlock_diagnostic *diagnostic) {
  graph_layout layout;
  uint64_t *parameter_bases = NULL;
  patchblob_discovered_target *discovered = NULL;
  size_t invocation_count = 0;
  size_t discovered_group_count = 0;
  size_t index;
  npunlock_status status;

  *targets = NULL;
  *target_count = 0;
  *group_count = 0;
  memset(&layout, 0, sizeof(layout));
  status = parse_graph(graph_blob, &layout, diagnostic);
  if (status != NPUNLOCK_STATUS_OK) {
    return status;
  }
  status = invocation_parameter_bases(&layout, &parameter_bases, &invocation_count, diagnostic);
  if (status != NPUNLOCK_STATUS_OK) {
    graph_layout_release(&layout);
    return status;
  }
  if (invocation_count == 0 || invocation_count > UINT32_MAX ||
      layout.sections[layout.ranges_index].size / ACT_RANGE_SIZE > UINT32_MAX) {
    status = unsupported(diagnostic, "ACT carrier has no discoverable invocation sequence");
    goto fail;
  }
  discovered = (patchblob_discovered_target *)calloc(invocation_count, sizeof(*discovered));
  if (discovered == NULL) {
    status = npunlock_set_diagnostic(diagnostic, NPUNLOCK_STATUS_OUT_OF_MEMORY,
                                     "patchblob.discover", "could not allocate target metadata");
    goto fail;
  }
  for (index = 0; index < invocation_count; ++index) {
    const graph_section *invocations = &layout.sections[layout.invocations_index];
    const graph_section *params = &layout.sections[layout.params_index];
    const uint8_t *record = layout.blob.data + invocations->offset + index * ACT_INVOCATION_SIZE;
    patchblob_target *target = &discovered[index].target;
    npunlock_patch_detail detail;
    graph_relocation range_relocation;
    memref_contract first_input;
    uint64_t base = parameter_bases[index];
    uint64_t limit =
        invocation_parameter_limit(parameter_bases, invocation_count, base, params->size);
    size_t record_count;
    size_t other;

    status = count_data_pointer_relocations(&layout, base, limit, &record_count, diagnostic);
    if (status != NPUNLOCK_STATUS_OK) {
      goto fail;
    }
    if (record_count < 2u || record_count > 9u || limit - base < record_count * MEMREF_SIZE) {
      status = unsupported(diagnostic,
                           "ACT invocation does not have a discoverable input/output contract");
      goto fail;
    }
    status = validate_memref(&layout, base, &first_input, diagnostic);
    if (status != NPUNLOCK_STATUS_OK) {
      goto fail;
    }
    discovered[index].struct_size = (uint32_t)sizeof(discovered[index]);
    target->struct_size = (uint32_t)sizeof(*target);
    target->invocation_index = (uint32_t)index;
    target->range_index = read_u32(record);
    target->expected_input_count = (uint32_t)(record_count - 1u);
    target->expected_element_count = first_input.element_count;
    target->expected_span_bytes = first_input.span_bytes;
    target->required_contract_flags =
        PATCHBLOB_CONTRACT_STATIC | PATCHBLOB_CONTRACT_DENSE | PATCHBLOB_CONTRACT_CMX |
        PATCHBLOB_CONTRACT_DISJOINT_OUTPUT |
        (first_input.dtype_kind == NPUNLOCK_DTYPE_F16 ? PATCHBLOB_CONTRACT_FP16
                                                      : PATCHBLOB_CONTRACT_FP32);
    status =
        validate_target(&layout, parameter_bases, invocation_count, target, &detail, diagnostic);
    if (status != NPUNLOCK_STATUS_OK) {
      goto fail;
    }
    status = find_range_relocation(&layout, target->range_index, &range_relocation, diagnostic);
    if (status != NPUNLOCK_STATUS_OK) {
      goto fail;
    }
    for (other = 0; other < index; ++other) {
      if (discovered[other].target.range_index == target->range_index) {
        status = unsupported(diagnostic, "ACT discovery found a range used more than once");
        goto fail;
      }
    }

    if (index == 0 || !invocation_group_identity_equal(&layout, index - 1u, index)) {
      for (other = 0; other < index; ++other) {
        if ((other == 0 || discovered[other - 1u].group_index != discovered[other].group_index) &&
            invocation_group_identity_equal(&layout, other, index)) {
          status = unsupported(diagnostic, "ACT group identity is non-contiguous and ambiguous");
          goto fail;
        }
      }
      if (discovered_group_count == UINT32_MAX) {
        status = unsupported(diagnostic, "ACT group count exceeds the supported index range");
        goto fail;
      }
      ++discovered_group_count;
    } else {
      const patchblob_target *previous = &discovered[index - 1u].target;
      if (previous->expected_input_count != target->expected_input_count ||
          previous->expected_element_count != target->expected_element_count ||
          previous->expected_span_bytes != target->expected_span_bytes ||
          previous->required_contract_flags != target->required_contract_flags) {
        status = unsupported(diagnostic, "ACT group contains inconsistent tensor contracts");
        goto fail;
      }
    }
    discovered[index].group_index = (uint32_t)(discovered_group_count - 1u);
  }

  free(parameter_bases);
  graph_layout_release(&layout);
  *targets = discovered;
  *target_count = invocation_count;
  *group_count = discovered_group_count;
  return NPUNLOCK_STATUS_OK;

fail:
  free(discovered);
  free(parameter_bases);
  graph_layout_release(&layout);
  return status;
}

static bool old_byte_may_change(const graph_layout *layout, const npunlock_patch_summary *summary,
                                size_t offset) {
  size_t index;
  if (offset >= 0x28u && offset < 0x30u) {
    return true;
  }
  for (index = 0; index < layout->section_count; ++index) {
    size_t header = layout->sections[index].header_offset;
    if ((offset >= header + 0x18u && offset < header + 0x20u) ||
        (index == layout->kernel_text_index && offset >= header + 0x20u &&
         offset < header + 0x28u)) {
      return true;
    }
  }
  for (index = 0; index < summary->detail_count; ++index) {
    size_t extent = summary->details[index].extent_file_offset;
    size_t addend = summary->details[index].addend_file_offset;
    if (extent >= summary->insertion_file_offset + summary->inserted_size) {
      extent -= summary->inserted_size;
    }
    if (addend >= summary->insertion_file_offset + summary->inserted_size) {
      addend -= summary->inserted_size;
    }
    if ((offset >= extent && offset < extent + 4u) || (offset >= addend && offset < addend + 8u)) {
      return true;
    }
  }
  return false;
}

static npunlock_status verify_preservation(const graph_layout *old_layout,
                                           const graph_layout *new_layout, npunlock_view image,
                                           npunlock_patch_summary *summary,
                                           npunlock_diagnostic *diagnostic) {
  size_t index;
  const graph_section *old_text = &old_layout->sections[old_layout->kernel_text_index];
  const graph_section *new_text = &new_layout->sections[new_layout->kernel_text_index];
  if (new_layout->section_count != old_layout->section_count ||
      new_text->size != old_text->size + summary->inserted_size ||
      memcmp(old_layout->blob.data + old_text->offset, new_layout->blob.data + new_text->offset,
             old_text->size) != 0) {
    return npunlock_set_diagnostic(diagnostic, NPUNLOCK_STATUS_INTERNAL_ERROR, "patchblob.verify",
                                   "KernelText preservation check failed");
  }
  for (index = 0; index < old_layout->section_count; ++index) {
    const graph_section *old_section = &old_layout->sections[index];
    const graph_section *new_section = &new_layout->sections[index];
    size_t expected_offset =
        old_section->offset >= summary->insertion_file_offset && old_section->offset != 0
            ? old_section->offset + summary->inserted_size
            : old_section->offset;
    if (old_section->type != new_section->type || old_section->flags != new_section->flags ||
        old_section->address != new_section->address || old_section->link != new_section->link ||
        old_section->info != new_section->info ||
        old_section->alignment != new_section->alignment ||
        old_section->entry_size != new_section->entry_size ||
        new_section->offset != expected_offset ||
        (index != old_layout->kernel_text_index && old_section->size != new_section->size)) {
      return npunlock_set_diagnostic(diagnostic, NPUNLOCK_STATUS_INTERNAL_ERROR, "patchblob.verify",
                                     "graph section metadata changed unexpectedly");
    }
    if (index == old_layout->kernel_text_index || index == old_layout->ranges_index ||
        index == old_layout->range_relocations_index || old_section->type == SHT_NOBITS) {
      continue;
    }
    if (old_section->size != new_section->size ||
        memcmp(old_layout->blob.data + old_section->offset,
               new_layout->blob.data + new_section->offset, old_section->size) != 0) {
      return npunlock_set_diagnostic(diagnostic, NPUNLOCK_STATUS_INTERNAL_ERROR, "patchblob.verify",
                                     "an unmodified graph section changed unexpectedly");
    }
  }
  if (memcmp(new_layout->blob.data + new_text->offset + summary->image_base, image.data,
             image.size) != 0) {
    return npunlock_set_diagnostic(diagnostic, NPUNLOCK_STATUS_INTERNAL_ERROR, "patchblob.verify",
                                   "appended code image does not match the ELF");
  }
  for (index = old_text->size; index < new_text->size; ++index) {
    if (index >= summary->image_base && index < summary->image_base + image.size) {
      continue;
    }
    if (new_layout->blob.data[new_text->offset + index] != 0) {
      return npunlock_set_diagnostic(diagnostic, NPUNLOCK_STATUS_INTERNAL_ERROR, "patchblob.verify",
                                     "appended alignment or tail padding is nonzero");
    }
  }
  for (index = 0; index < old_layout->blob.size; ++index) {
    size_t mapped =
        index >= summary->insertion_file_offset ? index + summary->inserted_size : index;
    if (!old_byte_may_change(old_layout, summary, index) &&
        old_layout->blob.data[index] != new_layout->blob.data[mapped]) {
      return npunlock_set_diagnostic(diagnostic, NPUNLOCK_STATUS_INTERNAL_ERROR, "patchblob.verify",
                                     "a byte outside the mutation set changed");
    }
  }
  summary->section_contents_preserved = true;
  summary->mutation_set_preserved = true;
  return NPUNLOCK_STATUS_OK;
}

void npunlock_patch_summary_release(npunlock_patch_summary *summary) {
  if (summary == NULL) {
    return;
  }
  free(summary->details);
  memset(summary, 0, sizeof(*summary));
}

npunlock_status npunlock_patch_graph_blob(npunlock_view graph_blob, npunlock_view image,
                                          const patchblob_target *targets, size_t target_count,
                                          uint32_t image_alignment, uint32_t tail_padding,
                                          npunlock_buffer *output, npunlock_patch_summary *summary,
                                          npunlock_diagnostic *diagnostic) {
  graph_layout layout;
  graph_layout patched_layout;
  uint64_t *parameter_bases = NULL;
  size_t invocation_count = 0;
  const graph_section *text;
  size_t image_base;
  size_t image_end;
  size_t new_text_size;
  size_t inserted_size;
  size_t output_size;
  uint8_t *patched = NULL;
  size_t index;
  npunlock_status status;

  memset(output, 0, sizeof(*output));
  memset(summary, 0, sizeof(*summary));
  memset(&layout, 0, sizeof(layout));
  memset(&patched_layout, 0, sizeof(patched_layout));
  status = parse_graph(graph_blob, &layout, diagnostic);
  if (status != NPUNLOCK_STATUS_OK) {
    return status;
  }
  status = invocation_parameter_bases(&layout, &parameter_bases, &invocation_count, diagnostic);
  if (status != NPUNLOCK_STATUS_OK) {
    graph_layout_release(&layout);
    return status;
  }
  summary->details = (npunlock_patch_detail *)calloc(target_count, sizeof(*summary->details));
  if (summary->details == NULL) {
    free(parameter_bases);
    graph_layout_release(&layout);
    return npunlock_set_diagnostic(diagnostic, NPUNLOCK_STATUS_OUT_OF_MEMORY, "patchblob.patch",
                                   "could not allocate patch details");
  }
  summary->detail_count = target_count;
  for (index = 0; index < target_count; ++index) {
    graph_relocation relocation;
    size_t other;
    for (other = 0; other < index; ++other) {
      if (targets[other].invocation_index == targets[index].invocation_index ||
          targets[other].range_index == targets[index].range_index) {
        status = unsupported(diagnostic, "target invocation and range selections must be unique");
        goto fail;
      }
    }
    status = validate_target(&layout, parameter_bases, invocation_count, &targets[index],
                             &summary->details[index], diagnostic);
    if (status != NPUNLOCK_STATUS_OK) {
      goto fail;
    }
    status = find_range_relocation(&layout, targets[index].range_index, &relocation, diagnostic);
    if (status != NPUNLOCK_STATUS_OK) {
      goto fail;
    }
    summary->details[index].addend_file_offset = relocation.file_offset + 16u;
    summary->details[index].old_addend = relocation.addend;
  }
  text = &layout.sections[layout.kernel_text_index];
  summary->insertion_file_offset = text->offset + text->size;
  for (index = 0; index < layout.section_count; ++index) {
    const graph_section *section = &layout.sections[index];
    size_t section_end;
    if (index == layout.kernel_text_index || section->type == SHT_NOBITS || section->size == 0) {
      continue;
    }
    if (!npunlock_checked_add_size(section->offset, section->size, &section_end) ||
        (section->offset < summary->insertion_file_offset &&
         summary->insertion_file_offset < section_end)) {
      status = malformed(diagnostic, "KernelText insertion point crosses another graph section");
      goto fail;
    }
  }
  {
    size_t section_table_size;
    size_t section_table_end;
    if (!npunlock_checked_mul_size(layout.section_count, ELF64_SECTION_SIZE, &section_table_size) ||
        !npunlock_checked_add_size(layout.section_table_offset, section_table_size,
                                   &section_table_end) ||
        (layout.section_table_offset < summary->insertion_file_offset &&
         summary->insertion_file_offset < section_table_end)) {
      status = malformed(diagnostic, "KernelText insertion point crosses the section table");
      goto fail;
    }
  }
  if (!align_up(text->size, image_alignment, &image_base) ||
      !npunlock_checked_add_size(image_base, image.size, &image_end) ||
      !npunlock_checked_add_size(image_end, tail_padding, &new_text_size) ||
      new_text_size < text->size) {
    status = npunlock_set_diagnostic(diagnostic, NPUNLOCK_STATUS_OVERFLOW, "patchblob.patch",
                                     "appended code image size overflows");
    goto fail;
  }
  inserted_size = new_text_size - text->size;
  if (!npunlock_checked_add_size(graph_blob.size, inserted_size, &output_size)) {
    status = npunlock_set_diagnostic(diagnostic, NPUNLOCK_STATUS_OVERFLOW, "patchblob.patch",
                                     "patched graph size overflows");
    goto fail;
  }
  patched = (uint8_t *)calloc(output_size, 1);
  if (patched == NULL) {
    status = npunlock_set_diagnostic(diagnostic, NPUNLOCK_STATUS_OUT_OF_MEMORY, "patchblob.patch",
                                     "could not allocate patched graph blob");
    goto fail;
  }
  summary->inserted_size = inserted_size;
  summary->image_base = image_base;
  summary->image_size = image.size;
  summary->old_section_table_offset = layout.section_table_offset;
  summary->new_section_table_offset = layout.section_table_offset >= summary->insertion_file_offset
                                          ? layout.section_table_offset + inserted_size
                                          : layout.section_table_offset;
  memcpy(patched, graph_blob.data, summary->insertion_file_offset);
  memcpy(patched + summary->insertion_file_offset + inserted_size,
         graph_blob.data + summary->insertion_file_offset,
         graph_blob.size - summary->insertion_file_offset);
  memcpy(patched + text->offset + image_base, image.data, image.size);
  write_u64(patched + 0x28, summary->new_section_table_offset);
  for (index = 0; index < layout.section_count; ++index) {
    const graph_section *section = &layout.sections[index];
    size_t header_offset = summary->new_section_table_offset + index * ELF64_SECTION_SIZE;
    size_t new_offset = section->offset >= summary->insertion_file_offset && section->offset != 0
                            ? section->offset + inserted_size
                            : section->offset;
    write_u64(patched + header_offset + 0x18, new_offset);
    if (index == layout.kernel_text_index) {
      write_u64(patched + header_offset + 0x20, new_text_size);
    }
  }
  for (index = 0; index < target_count; ++index) {
    npunlock_patch_detail *detail = &summary->details[index];
    size_t extent_offset = detail->extent_file_offset >= summary->insertion_file_offset
                               ? detail->extent_file_offset + inserted_size
                               : detail->extent_file_offset;
    size_t addend_offset = detail->addend_file_offset >= summary->insertion_file_offset
                               ? detail->addend_file_offset + inserted_size
                               : detail->addend_file_offset;
    if (image.size > UINT32_MAX || image_base > INT64_MAX) {
      status =
          unsupported(diagnostic, "appended code image does not fit the observed range fields");
      goto fail;
    }
    write_u32(patched + extent_offset, (uint32_t)image.size);
    write_u64(patched + addend_offset, image_base);
    detail->extent_file_offset = extent_offset;
    detail->addend_file_offset = addend_offset;
    detail->new_extent = (uint32_t)image.size;
    detail->new_addend = (int64_t)image_base;
  }
  status = parse_graph((npunlock_view){patched, output_size}, &patched_layout, diagnostic);
  if (status != NPUNLOCK_STATUS_OK) {
    goto fail;
  }
  status = verify_preservation(&layout, &patched_layout, image, summary, diagnostic);
  if (status != NPUNLOCK_STATUS_OK) {
    goto fail;
  }
  status = npunlock_buffer_adopt_malloc(patched, output_size, output);
  if (status != NPUNLOCK_STATUS_OK) {
    patched = NULL;
    goto fail;
  }
  patched = NULL;
  free(parameter_bases);
  graph_layout_release(&patched_layout);
  graph_layout_release(&layout);
  return NPUNLOCK_STATUS_OK;

fail:
  free(patched);
  free(parameter_bases);
  graph_layout_release(&patched_layout);
  graph_layout_release(&layout);
  npunlock_patch_summary_release(summary);
  return status;
}
