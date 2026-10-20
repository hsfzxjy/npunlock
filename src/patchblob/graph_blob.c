#include "graph_blob.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include "internal.h"

#define ELF64_HEADER_SIZE 64u
#define ELF64_SECTION_SIZE 64u
#define ELF64_SECTION_ALIGNMENT 8u
#define ELF64_SYMBOL_SIZE 24u
#define ELF64_RELA_SIZE 24u
#define ACT_INVOCATION_SIZE 0x40u
#define ACT_RANGE_SIZE 0x18u
#define MEMREF_SIZE 0x28u
#define DMA_TASK_SIZE 0x80u
#define DPU_INVARIANT_SIZE 0x130u
#define DPU_INPUT_ADDRESS_OFFSET 0x5cu
#define DPU_OUTPUT_ADDRESS_OFFSET 0xa0u
#define ACT_EDGE_MAX_COMPARISONS 16777216u
#define GRAPH_CODE_ADDRESS 0x1d000000u
#define SHT_NULL 0u
#define SHT_PROGBITS 1u
#define SHT_SYMTAB 2u
#define SHT_STRTAB 3u
#define SHT_RELA 4u
#define SHT_NOTE 7u
#define SHT_NOBITS 8u
#define SHT_VPU_PLATFORM_INFO 0x8aaaaaafu
#define SHT_VPU_COMPATIBILITY 0x8aaaaab2u
#define R_VPU_64 0u
#define R_VPU_64_LSHIFT 3u
#define R_VPU_32 4u
#define R_VPU_32_RTM 5u
#define R_VPU_32_MULTICAST_BASE 7u
#define VPU_RT_SYMTAB 0xff20u
#define VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR 0u
#define VPU_NNRD_SYM_LAST 7u
#define NPU_VERSION_NOTE_SIZE 0x20u
#define NPU_PLATFORM_INFO_SIZE 0x28u

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
  size_t compatibility_index;
  size_t loader_abi_version_index;
  size_t mapped_inference_version_index;
  size_t platform_info_index;
  uint32_t tile_count;
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
  uint32_t rank;
  int32_t dimensions[NPUNLOCK_ROUTING_MAX_RANK];
  int64_t strides_bits[NPUNLOCK_ROUTING_MAX_RANK];
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

static bool is_power_of_two(size_t value) { return value != 0u && (value & (value - 1u)) == 0u; }

static bool ranges_overlap(size_t left_offset, size_t left_size, size_t right_offset,
                           size_t right_size) {
  size_t left_end;
  size_t right_end;
  return left_size != 0u && right_size != 0u &&
         npunlock_checked_add_size(left_offset, left_size, &left_end) &&
         npunlock_checked_add_size(right_offset, right_size, &right_end) &&
         left_offset < right_end && right_offset < left_end;
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

static bool token_starts_with(const uint8_t *token, size_t token_size, const char *prefix) {
  size_t prefix_size = strlen(prefix);
  return token_size >= prefix_size && memcmp(token, prefix, prefix_size) == 0;
}

static bool token_equals(const uint8_t *token, size_t token_size, const char *expected) {
  size_t expected_size = strlen(expected);
  return token_size == expected_size && memcmp(token, expected, expected_size) == 0;
}

static bool name_has_decimal_suffix(const char *name, const char *prefix) {
  size_t name_size = strlen(name);
  size_t prefix_size = strlen(prefix);
  const char *suffix;
  if (name_size <= prefix_size || memcmp(name, prefix, prefix_size) != 0) {
    return false;
  }
  suffix = name + prefix_size;
  if (*suffix == '\0') {
    return false;
  }
  while (*suffix != '\0') {
    if (*suffix < '0' || *suffix > '9') {
      return false;
    }
    ++suffix;
  }
  return true;
}

static npunlock_status validate_graph_compatibility(graph_layout *layout,
                                                    npunlock_diagnostic *diagnostic) {
  const graph_section *section = &layout->sections[layout->compatibility_index];
  const uint8_t *data;
  size_t data_size;
  size_t offset = 0;
  bool compiler_found = false;
  bool npu_found = false;
  bool tiles_found = false;
  bool loader_elf_found = false;
  bool mapped_inference_found = false;

  if (section->type != SHT_VPU_COMPATIBILITY || section->flags != 0u || section->address != 0u ||
      section->link != 0u || section->info != 0u || section->alignment != 1u ||
      section->entry_size != 1u || section->size == 0u) {
    return unsupported(diagnostic, "graph compatibility_string section has an unsupported form");
  }
  data = layout->blob.data + section->offset;
  data_size = section->size;
  if (data[data_size - 1u] == 0u) {
    --data_size;
  }
  if (data_size == 0u) {
    return malformed(diagnostic, "graph compatibility_string is empty");
  }
  while (offset < data_size) {
    const uint8_t *token = data + offset;
    const uint8_t *separator = memchr(token, ';', data_size - offset);
    size_t token_size = separator == NULL ? data_size - offset : (size_t)(separator - token);

    if (token_size == 0u || memchr(token, 0, token_size) != NULL) {
      return malformed(diagnostic, "graph compatibility_string contains a malformed field");
    }
    if (token_starts_with(token, token_size, "compiler=")) {
      if (compiler_found) {
        return malformed(diagnostic,
                         "graph compatibility_string contains duplicate compiler fields");
      }
      compiler_found = true;
      if (!token_equals(token, token_size, "compiler=8.3")) {
        return unsupported(diagnostic, "graph compiler version is not the supported 8.3 contract");
      }
    } else if (token_starts_with(token, token_size, "npu=")) {
      if (npu_found) {
        return malformed(diagnostic, "graph compatibility_string contains duplicate npu fields");
      }
      npu_found = true;
      if (!token_equals(token, token_size, "npu=3720")) {
        return unsupported(diagnostic, "graph target is not the supported NPU3720 contract");
      }
    } else if (token_starts_with(token, token_size, "t=")) {
      if (tiles_found) {
        return malformed(diagnostic, "graph compatibility_string contains duplicate t fields");
      }
      tiles_found = true;
      if (token_equals(token, token_size, "t=1")) {
        layout->tile_count = 1u;
      } else if (token_equals(token, token_size, "t=2")) {
        layout->tile_count = 2u;
      } else {
        return unsupported(diagnostic,
                           "graph tile count is outside the supported one- or two-tile contract");
      }
    } else if (token_starts_with(token, token_size, "elf=")) {
      if (loader_elf_found) {
        return malformed(diagnostic, "graph compatibility_string contains duplicate elf fields");
      }
      loader_elf_found = true;
      if (!token_equals(token, token_size, "elf=1.2.0")) {
        return unsupported(diagnostic, "graph loader ELF version is not the supported 1.2.0");
      }
    } else if (token_starts_with(token, token_size, "mi=")) {
      if (mapped_inference_found) {
        return malformed(diagnostic, "graph compatibility_string contains duplicate mi fields");
      }
      mapped_inference_found = true;
      if (!token_equals(token, token_size, "mi=7.0.4")) {
        return unsupported(diagnostic, "graph mapped-inference version is not the supported 7.0.4");
      }
    }
    if (separator == NULL) {
      offset = data_size;
    } else {
      offset += token_size + 1u;
    }
  }
  if (!compiler_found || !npu_found || !tiles_found || !loader_elf_found ||
      !mapped_inference_found) {
    return unsupported(diagnostic,
                       "graph compatibility_string lacks a required compatibility field");
  }
  return NPUNLOCK_STATUS_OK;
}

static npunlock_status validate_version_note(const graph_layout *layout, size_t section_index,
                                             const char owner[4], uint32_t note_type,
                                             const uint32_t expected[4], const char *message,
                                             npunlock_diagnostic *diagnostic) {
  const graph_section *section = &layout->sections[section_index];
  const uint8_t *note;
  size_t index;

  if (section->type != SHT_NOTE || section->flags != 0u || section->address != 0u ||
      section->link != 0u || section->info != 0u || section->alignment != 4u ||
      section->entry_size != 1u || section->size != NPU_VERSION_NOTE_SIZE) {
    return unsupported(diagnostic, message);
  }
  note = layout->blob.data + section->offset;
  if (read_u32(note) != 4u || read_u32(note + 4u) != 16u || read_u32(note + 8u) != note_type ||
      memcmp(note + 12u, owner, 4u) != 0) {
    return unsupported(diagnostic, message);
  }
  for (index = 0; index < 4u; ++index) {
    if (read_u32(note + 16u + index * 4u) != expected[index]) {
      return unsupported(diagnostic, message);
    }
  }
  return NPUNLOCK_STATUS_OK;
}

static npunlock_status validate_act_tile_indices(const graph_layout *layout,
                                                 npunlock_diagnostic *diagnostic) {
  const graph_section *invocations = &layout->sections[layout->invocations_index];
  size_t index;
  for (index = 0; index < invocations->size / ACT_INVOCATION_SIZE; ++index) {
    const uint8_t *record = layout->blob.data + invocations->offset + index * ACT_INVOCATION_SIZE;
    if (read_u32(record + 0x34u) >= layout->tile_count) {
      return unsupported(diagnostic,
                         "ACT invocation tile index exceeds the graph compatibility contract");
    }
  }
  return NPUNLOCK_STATUS_OK;
}

static npunlock_status validate_platform_info(const graph_layout *layout,
                                              npunlock_diagnostic *diagnostic) {
  const graph_section *section = &layout->sections[layout->platform_info_index];
  const uint8_t *data;

  if (section->type != SHT_VPU_PLATFORM_INFO || section->flags != 0u || section->address != 0u ||
      section->link != 0u || section->info != 0u || section->size != NPU_PLATFORM_INFO_SIZE ||
      section->alignment != 8u || section->entry_size != 1u) {
    return unsupported(diagnostic, "graph platform-info section has an unsupported form");
  }
  data = layout->blob.data + section->offset;
  if (read_u64(data) != 0x20u || read_u64(data + 8u) != 0u || read_u64(data + 16u) != 1u ||
      read_u64(data + 24u) != 8u || read_u64(data + 32u) != 3u) {
    return unsupported(diagnostic,
                       "graph platform-info record is not the supported NPU3720 contract");
  }
  return NPUNLOCK_STATUS_OK;
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
  layout->compatibility_index = SIZE_MAX;
  layout->loader_abi_version_index = SIZE_MAX;
  layout->mapped_inference_version_index = SIZE_MAX;
  layout->platform_info_index = SIZE_MAX;
  if (blob.size < ELF64_HEADER_SIZE ||
      memcmp(bytes,
             "\x7f"
             "ELF",
             4) != 0 ||
      bytes[4] != 2 || bytes[5] != 1) {
    return malformed(diagnostic, "graph blob is not a supported ELF64 little-endian image");
  }
  if (bytes[6] != 0u || bytes[7] != 0u || bytes[8] != 0u || read_u16(bytes + 0x10) != 1u ||
      read_u16(bytes + 0x12) != 0u || read_u32(bytes + 0x14) != 0u ||
      read_u64(bytes + 0x18) != 0u || read_u32(bytes + 0x30) != 0u) {
    return unsupported(diagnostic, "graph ELF identity is not the supported NPU native contract");
  }
  if (read_u64(bytes + 0x20) != 0u || read_u16(bytes + 0x34) != ELF64_HEADER_SIZE ||
      read_u16(bytes + 0x36) != 0u || read_u16(bytes + 0x38) != 0u ||
      read_u16(bytes + 0x3a) != ELF64_SECTION_SIZE) {
    return unsupported(diagnostic,
                       "graph ELF header layout is not the supported section-only contract");
  }
  section_table_u64 = read_u64(bytes + 0x28);
  layout->section_count = read_u16(bytes + 0x3c);
  layout->section_name_index = read_u16(bytes + 0x3e);
  if (!u64_to_size(section_table_u64, &layout->section_table_offset) ||
      layout->section_table_offset != ELF64_HEADER_SIZE || layout->section_count == 0 ||
      layout->section_name_index >= layout->section_count ||
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
        (section->alignment != 0u && !is_power_of_two(section->alignment)) ||
        (section->alignment > 1u && section->offset % section->alignment != 0u) ||
        (section->type != SHT_NOBITS &&
         !range_in_bounds(section->offset, section->size, blob.size))) {
      graph_layout_release(layout);
      return malformed(diagnostic, "graph section extent or alignment is malformed");
    }
  }
  {
    const graph_section *null_section = &layout->sections[0];
    const uint8_t *null_header = bytes + null_section->header_offset;
    size_t left;
    size_t right;
    if (read_u32(null_header) != 0u || null_section->type != SHT_NULL ||
        null_section->flags != 0u || null_section->address != 0u || null_section->offset != 0u ||
        null_section->size != 0u || null_section->link != 0u || null_section->info != 0u ||
        null_section->alignment != 4u || null_section->entry_size != 0u) {
      graph_layout_release(layout);
      return unsupported(diagnostic, "graph null section is not the supported native form");
    }
    for (left = 1u; left < layout->section_count; ++left) {
      const graph_section *left_section = &layout->sections[left];
      if (left_section->type == SHT_NOBITS || left_section->size == 0u) {
        continue;
      }
      if (ranges_overlap(layout->section_table_offset, table_size, left_section->offset,
                         left_section->size)) {
        graph_layout_release(layout);
        return malformed(diagnostic, "graph section overlaps the section table");
      }
      for (right = left + 1u; right < layout->section_count; ++right) {
        const graph_section *right_section = &layout->sections[right];
        if (right_section->type != SHT_NOBITS &&
            ranges_overlap(left_section->offset, left_section->size, right_section->offset,
                           right_section->size)) {
          graph_layout_release(layout);
          return malformed(diagnostic, "graph file-backed sections overlap");
        }
      }
    }
  }
  string_offset = layout->sections[layout->section_name_index].offset;
  string_size = layout->sections[layout->section_name_index].size;
  if (layout->section_name_index != 1u ||
      layout->sections[layout->section_name_index].type != SHT_STRTAB ||
      layout->sections[layout->section_name_index].flags != 0u ||
      layout->sections[layout->section_name_index].address != 0u ||
      layout->sections[layout->section_name_index].link != 0u ||
      layout->sections[layout->section_name_index].info != 0u ||
      layout->sections[layout->section_name_index].alignment != 4u ||
      layout->sections[layout->section_name_index].entry_size != 0u || string_size == 0u ||
      bytes[string_offset] != 0u) {
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
    MATCH_REQUIRED(compatibility_index, "compatibility_string")
    MATCH_REQUIRED(loader_abi_version_index, ".note.LoaderABIVersion")
    MATCH_REQUIRED(mapped_inference_version_index, ".note.MappedInferenceVersion")
    MATCH_REQUIRED(platform_info_index, ".meta.PlatformInfo")
#undef MATCH_REQUIRED
  }
  if (strcmp(layout->sections[layout->section_name_index].name, ".strtab") != 0) {
    graph_layout_release(layout);
    return unsupported(diagnostic, "graph section-name table is not the supported native form");
  }
  if (layout->kernel_text_index == SIZE_MAX || layout->params_index == SIZE_MAX ||
      layout->invocations_index == SIZE_MAX || layout->ranges_index == SIZE_MAX ||
      layout->range_relocations_index == SIZE_MAX) {
    graph_layout_release(layout);
    return unsupported(diagnostic, "graph does not contain the required ACT carrier sections");
  }
  if (layout->compatibility_index == SIZE_MAX) {
    graph_layout_release(layout);
    return unsupported(diagnostic, "graph does not contain a compatibility_string section");
  }
  if (layout->loader_abi_version_index == SIZE_MAX ||
      layout->mapped_inference_version_index == SIZE_MAX) {
    graph_layout_release(layout);
    return unsupported(diagnostic, "graph does not contain the required version-note sections");
  }
  if (layout->platform_info_index == SIZE_MAX) {
    graph_layout_release(layout);
    return unsupported(diagnostic, "graph does not contain a platform-info section");
  }
  if (layout->sections[layout->kernel_text_index].type != SHT_PROGBITS ||
      layout->sections[layout->params_index].type != SHT_PROGBITS ||
      layout->sections[layout->invocations_index].type != SHT_PROGBITS ||
      layout->sections[layout->ranges_index].type != SHT_PROGBITS ||
      layout->sections[layout->range_relocations_index].type != SHT_RELA) {
    graph_layout_release(layout);
    return unsupported(diagnostic, "required ACT carrier sections have unsupported types");
  }
  if (layout->sections[layout->kernel_text_index].flags != 0x40000000u ||
      layout->sections[layout->kernel_text_index].address != 0u ||
      layout->sections[layout->kernel_text_index].link != 0u ||
      layout->sections[layout->kernel_text_index].info != 0u ||
      layout->sections[layout->kernel_text_index].alignment != 1024u ||
      layout->sections[layout->kernel_text_index].entry_size != 1u ||
      layout->sections[layout->params_index].flags != 0x40000000u ||
      layout->sections[layout->params_index].address != 0u ||
      layout->sections[layout->params_index].link != 0u ||
      layout->sections[layout->params_index].info != 0u ||
      layout->sections[layout->params_index].alignment != 64u ||
      layout->sections[layout->params_index].entry_size != 1u ||
      layout->sections[layout->invocations_index].flags != 0x20000004u ||
      layout->sections[layout->invocations_index].address != 0u ||
      layout->sections[layout->invocations_index].link != 0u ||
      layout->sections[layout->invocations_index].info != 0u ||
      layout->sections[layout->invocations_index].alignment != 64u ||
      layout->sections[layout->invocations_index].entry_size != 1u ||
      layout->sections[layout->ranges_index].flags != 0x20000004u ||
      layout->sections[layout->ranges_index].address != 0u ||
      layout->sections[layout->ranges_index].link != 0u ||
      layout->sections[layout->ranges_index].info != 0u ||
      layout->sections[layout->ranges_index].alignment != 64u ||
      layout->sections[layout->ranges_index].entry_size != 1u ||
      layout->sections[layout->range_relocations_index].flags != 0x40u ||
      layout->sections[layout->range_relocations_index].address != 0u ||
      layout->sections[layout->range_relocations_index].alignment != 8u) {
    graph_layout_release(layout);
    return unsupported(diagnostic, "required ACT carrier sections have unsupported metadata");
  }
  if (layout->sections[layout->invocations_index].size % ACT_INVOCATION_SIZE != 0 ||
      layout->sections[layout->ranges_index].size % ACT_RANGE_SIZE != 0 ||
      layout->sections[layout->range_relocations_index].entry_size != ELF64_RELA_SIZE ||
      layout->sections[layout->range_relocations_index].size % ELF64_RELA_SIZE != 0) {
    graph_layout_release(layout);
    return malformed(diagnostic, "ACT carrier section sizes are malformed");
  }
  {
    static const uint32_t loader_abi_version[4] = {0u, 1u, 2u, 0u};
    static const uint32_t mapped_inference_version[4] = {0u, 7u, 0u, 4u};
    npunlock_status status = validate_graph_compatibility(layout, diagnostic);
    if (status == NPUNLOCK_STATUS_OK) {
      status = validate_version_note(
          layout, layout->loader_abi_version_index, "GNU", 1u, loader_abi_version,
          "graph loader ABI note is not the supported 1.2.0 contract", diagnostic);
    }
    if (status == NPUNLOCK_STATUS_OK) {
      status = validate_version_note(
          layout, layout->mapped_inference_version_index, "MIV", 0xa000u, mapped_inference_version,
          "graph mapped-inference note is not the supported 7.0.4 contract", diagnostic);
    }
    if (status == NPUNLOCK_STATUS_OK) {
      status = validate_act_tile_indices(layout, diagnostic);
    }
    if (status == NPUNLOCK_STATUS_OK) {
      status = validate_platform_info(layout, diagnostic);
    }
    if (status != NPUNLOCK_STATUS_OK) {
      graph_layout_release(layout);
      return status;
    }
  }
  return NPUNLOCK_STATUS_OK;
}

static npunlock_status resolve_relocation(const graph_layout *layout, size_t relocation_index,
                                          size_t entry_index, graph_relocation *relocation,
                                          npunlock_diagnostic *diagnostic) {
  const graph_section *relocations;
  const graph_section *target;
  const graph_section *symbols;
  const graph_section *strings;
  const uint8_t *entry;
  uint64_t info;
  size_t symbol_count;
  const uint8_t *symbol;
  uint32_t symbol_name_offset;
  uint16_t source_index;
  const char *expected_relocation_name;
  const char *expected_symbol_name;
  const char *symbol_name;
  size_t write_offset;
  size_t write_size;

  if (relocation_index >= layout->section_count) {
    return malformed(diagnostic, "relocation section index is invalid");
  }
  relocations = &layout->sections[relocation_index];
  if (relocations->type != SHT_RELA || relocations->entry_size != ELF64_RELA_SIZE ||
      relocations->size % ELF64_RELA_SIZE != 0 ||
      entry_index >= relocations->size / ELF64_RELA_SIZE) {
    return malformed(diagnostic, "relocation entry is malformed");
  }
  if (relocations->flags != 0x40u || relocations->address != 0u || relocations->alignment != 8u ||
      relocations->info >= layout->section_count) {
    return unsupported(diagnostic, "relocation section has unsupported metadata");
  }
  target = &layout->sections[relocations->info];
  if (strcmp(target->name, ".text.KernelParams") == 0) {
    expected_relocation_name = ".rlt.text.KernelParams";
  } else if (strcmp(target->name, ".text.ActKernelInvocations") == 0) {
    expected_relocation_name = ".rlt.text.ActKernelInvocations";
  } else if (strcmp(target->name, ".text.ActKernelRanges") == 0) {
    expected_relocation_name = ".rlt.text.ActKernelRanges";
  } else {
    return unsupported(diagnostic, "relocation targets an unsupported native section");
  }
  if (strcmp(relocations->name, expected_relocation_name) != 0) {
    return unsupported(diagnostic, "relocation section name does not match its target");
  }
  relocation->file_offset = relocations->offset + entry_index * ELF64_RELA_SIZE;
  entry = layout->blob.data + relocation->file_offset;
  relocation->target_offset = read_u64(entry);
  info = read_u64(entry + 8);
  relocation->symbol = (uint32_t)(info >> 32);
  relocation->type = (uint32_t)info;
  relocation->addend = read_i64(entry + 16);
  switch (relocation->type) {
  case R_VPU_64_LSHIFT:
    write_size = 8u;
    break;
  case R_VPU_32:
  case R_VPU_32_RTM:
    write_size = 4u;
    break;
  default:
    return unsupported(diagnostic, "ACT relocation uses an unsupported relocation type");
  }
  if (!u64_to_size(relocation->target_offset, &write_offset) ||
      relocation->target_offset % write_size != 0u ||
      !range_in_bounds(write_offset, write_size, target->size)) {
    return malformed(diagnostic, "ACT relocation write is outside or misaligned in its target");
  }
  relocation->special_source = false;
  relocation->source_section_index = SIZE_MAX;
  if (relocations->link >= layout->section_count) {
    if (relocations->link != VPU_RT_SYMTAB || relocation->symbol > VPU_NNRD_SYM_LAST) {
      return unsupported(diagnostic, "relocation uses an unsupported VPU runtime symbol");
    }
    relocation->special_source = true;
    return NPUNLOCK_STATUS_OK;
  }
  symbols = &layout->sections[relocations->link];
  if (strcmp(symbols->name, ".symtab.tasks") != 0 || symbols->type != SHT_SYMTAB ||
      symbols->flags != 0u || symbols->address != 0u || symbols->info != 0u ||
      symbols->alignment != 8u || symbols->entry_size != ELF64_SYMBOL_SIZE ||
      symbols->size % ELF64_SYMBOL_SIZE != 0 || symbols->link >= layout->section_count) {
    return unsupported(diagnostic, "relocation symbol table is not the supported native form");
  }
  strings = &layout->sections[symbols->link];
  if (strcmp(strings->name, ".symstrtab") != 0 || strings->type != SHT_STRTAB ||
      strings->flags != 0u || strings->address != 0u || strings->link != 0u ||
      strings->info != 0u || strings->alignment != 4u || strings->entry_size != 0u ||
      strings->size == 0u || layout->blob.data[strings->offset] != 0u) {
    return unsupported(diagnostic, "relocation string table is not the supported native form");
  }
  symbol_count = symbols->size / ELF64_SYMBOL_SIZE;
  if (relocation->symbol >= symbol_count) {
    return malformed(diagnostic, "relocation symbol index is outside the symbol table");
  }
  symbol = layout->blob.data + symbols->offset + relocation->symbol * ELF64_SYMBOL_SIZE;
  symbol_name_offset = read_u32(symbol);
  source_index = read_u16(symbol + 6);
  if (symbol[4] != 0u || symbol[5] != 0u || read_u64(symbol + 8) != 0u ||
      read_u64(symbol + 16) != 0u || source_index >= layout->section_count) {
    return unsupported(diagnostic, "relocation uses an unsupported source symbol");
  }
  if (strcmp(layout->sections[source_index].name, ".text.KernelText") == 0) {
    expected_symbol_name = "sym_kernelTextSection";
  } else if (strcmp(layout->sections[source_index].name, ".text.KernelData") == 0) {
    expected_symbol_name = "sym_kernelDataSection";
  } else if (strcmp(layout->sections[source_index].name, ".text.KernelParams") == 0) {
    expected_symbol_name = "sym_kernelParamsSection";
  } else {
    return unsupported(diagnostic, "relocation references an unsupported source section");
  }
  if (symbol_name_offset >= strings->size) {
    return malformed(diagnostic, "relocation symbol name is outside the string table");
  }
  symbol_name = (const char *)(layout->blob.data + strings->offset + symbol_name_offset);
  if (memchr(symbol_name, 0, strings->size - symbol_name_offset) == NULL) {
    return malformed(diagnostic, "relocation symbol name is not terminated");
  }
  if (strcmp(symbol_name, expected_symbol_name) != 0) {
    return unsupported(diagnostic, "relocation symbol name does not match its source section");
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
      if (candidate.target_offset == target_offset && candidate.type == R_VPU_32 &&
          ((require_special && candidate.special_source &&
            candidate.symbol == VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR) ||
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
  NPUNLOCK_DTYPE_U32 = 5,
  NPUNLOCK_DTYPE_U16 = 6,
  NPUNLOCK_DTYPE_U8 = 7,
  NPUNLOCK_DTYPE_I32 = 9,
  NPUNLOCK_DTYPE_I16 = 10,
  NPUNLOCK_DTYPE_I8 = 11,
  NPUNLOCK_DTYPE_MAX = 12,
};

struct npunlock_dtype_info {
  uint32_t stride_bits;
  uint32_t size_bytes;
};

static const struct npunlock_dtype_info dtype_table[NPUNLOCK_DTYPE_MAX] = {
    [NPUNLOCK_DTYPE_F32] = {.stride_bits = 32, .size_bytes = 4},
    [NPUNLOCK_DTYPE_F16] = {.stride_bits = 16, .size_bytes = 2},
    [NPUNLOCK_DTYPE_U32] = {.stride_bits = 32, .size_bytes = 4},
    [NPUNLOCK_DTYPE_U16] = {.stride_bits = 16, .size_bytes = 2},
    [NPUNLOCK_DTYPE_U8] = {.stride_bits = 8, .size_bytes = 1},
    [NPUNLOCK_DTYPE_I32] = {.stride_bits = 32, .size_bytes = 4},
    [NPUNLOCK_DTYPE_I16] = {.stride_bits = 16, .size_bytes = 2},
    [NPUNLOCK_DTYPE_I8] = {.stride_bits = 8, .size_bytes = 1},
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
    return unsupported(diagnostic,
                       "selected invocation is not static FP16/FP32/U8/U16/U32/I32/I16/I8 CMX");
  }
  expected_stride = dtype_table[type_kind].stride_bits;
  rank = read_u32(record + 0x0c);
  if (rank == 0 || rank > NPUNLOCK_ROUTING_MAX_RANK) {
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
  contract->rank = rank;
  memcpy(contract->dimensions, dimension_values, (size_t)rank * sizeof(dimension_values[0]));
  memcpy(contract->strides_bits, stride_values, (size_t)rank * sizeof(stride_values[0]));
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
      if (relocation.special_source && relocation.symbol == VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR &&
          relocation.type == R_VPU_32 && relocation.target_offset >= base &&
          relocation.target_offset < limit) {
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
  uint32_t scalar_flags = target->required_contract_flags & PATCHBLOB_CONTRACT_INPUT_1_SCALAR;
  uint32_t expected_dtype;
  if (target->invocation_index >= invocation_count ||
      target->range_index >= ranges->size / ACT_RANGE_SIZE || target->expected_input_count > 8u) {
    return unsupported(diagnostic, "selected invocation or range index is outside the ACT carrier");
  }
  if ((precision_flags != PATCHBLOB_CONTRACT_FP16 && precision_flags != PATCHBLOB_CONTRACT_FP32) ||
      target->required_contract_flags != (base_flags | precision_flags | scalar_flags) ||
      (scalar_flags != 0u &&
       (target->expected_input_count != 2u || target->expected_element_count <= 1u))) {
    return unsupported(diagnostic,
                       "target requires the observed static dense FP16/FP32 CMX contract");
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
    uint64_t expected_element_count = target->expected_element_count;
    uint64_t expected_span_bytes = target->expected_span_bytes;
    npunlock_status status =
        validate_memref(layout, base + (uint64_t)index * MEMREF_SIZE, &inputs[index], diagnostic);
    if (status != NPUNLOCK_STATUS_OK) {
      return status;
    }
    if (index == 1u && scalar_flags != 0u) {
      expected_element_count = 1u;
      expected_span_bytes = dtype_table[expected_dtype].size_bytes;
    }
    if (inputs[index].element_count != expected_element_count ||
        inputs[index].span_bytes != expected_span_bytes ||
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
  detail->contract_flags = target->required_contract_flags;
  detail->extent_file_offset = range_offset + 0x0cu;
  detail->old_extent = read_u32(layout->blob.data + detail->extent_file_offset);
  return NPUNLOCK_STATUS_OK;
}

static uint32_t public_precision_v2(uint32_t dtype_kind) {
  if (dtype_kind == NPUNLOCK_DTYPE_F16) {
    return PATCHBLOB_TENSOR_PRECISION_FP16_V2;
  }
  if (dtype_kind == NPUNLOCK_DTYPE_F32) {
    return PATCHBLOB_TENSOR_PRECISION_FP32_V2;
  }
  if (dtype_kind == NPUNLOCK_DTYPE_I32) {
    return PATCHBLOB_TENSOR_PRECISION_I32_V2;
  }
  if (dtype_kind == NPUNLOCK_DTYPE_I16) {
    return PATCHBLOB_TENSOR_PRECISION_I16_V2;
  }
  if (dtype_kind == NPUNLOCK_DTYPE_I8) {
    return PATCHBLOB_TENSOR_PRECISION_I8_V2;
  }
  if (dtype_kind == NPUNLOCK_DTYPE_U8) {
    return PATCHBLOB_TENSOR_PRECISION_U8_V2;
  }
  if (dtype_kind == NPUNLOCK_DTYPE_U16) {
    return PATCHBLOB_TENSOR_PRECISION_U16_V2;
  }
  if (dtype_kind == NPUNLOCK_DTYPE_U32) {
    return PATCHBLOB_TENSOR_PRECISION_U32_V2;
  }
  return 0u;
}

static uint32_t dtype_from_public_precision_v2(uint32_t precision) {
  if (precision == PATCHBLOB_TENSOR_PRECISION_FP16_V2) {
    return NPUNLOCK_DTYPE_F16;
  }
  if (precision == PATCHBLOB_TENSOR_PRECISION_FP32_V2) {
    return NPUNLOCK_DTYPE_F32;
  }
  if (precision == PATCHBLOB_TENSOR_PRECISION_I32_V2) {
    return NPUNLOCK_DTYPE_I32;
  }
  if (precision == PATCHBLOB_TENSOR_PRECISION_I16_V2) {
    return NPUNLOCK_DTYPE_I16;
  }
  if (precision == PATCHBLOB_TENSOR_PRECISION_I8_V2) {
    return NPUNLOCK_DTYPE_I8;
  }
  if (precision == PATCHBLOB_TENSOR_PRECISION_U8_V2) {
    return NPUNLOCK_DTYPE_U8;
  }
  if (precision == PATCHBLOB_TENSOR_PRECISION_U16_V2) {
    return NPUNLOCK_DTYPE_U16;
  }
  if (precision == PATCHBLOB_TENSOR_PRECISION_U32_V2) {
    return NPUNLOCK_DTYPE_U32;
  }
  return 0;
}

static npunlock_status validate_target_v2(const graph_layout *layout,
                                          const uint64_t *parameter_bases, size_t invocation_count,
                                          const patchblob_target_v2 *target,
                                          npunlock_patch_detail *detail,
                                          npunlock_diagnostic *diagnostic) {
  const graph_section *invocations = &layout->sections[layout->invocations_index];
  const graph_section *params = &layout->sections[layout->params_index];
  const graph_section *ranges = &layout->sections[layout->ranges_index];
  memref_contract actual[PATCHBLOB_TARGET_V2_MAX_TENSORS];
  const patchblob_tensor_contract_v2 *output;
  uint64_t base;
  uint64_t limit;
  size_t invocation_offset;
  size_t range_offset;
  size_t input_count;
  size_t scalar_input_count = 0u;
  size_t index;
  bool mixed_precision = false;
  bool supported_unary_conversion;
  bool supported_multi_scalar_fp16;
  bool supported_hybrid_scalar_fp16;
  const uint32_t input_flags =
      PATCHBLOB_TENSOR_STATIC_V2 | PATCHBLOB_TENSOR_DENSE_V2 | PATCHBLOB_TENSOR_CMX_V2;
  const uint32_t output_flags = input_flags | PATCHBLOB_TENSOR_DISJOINT_FROM_INPUTS_V2;

  if (target->abi_version != PATCHBLOB_TARGET_V2_ABI_VERSION || target->tensor_count < 2u ||
      target->tensor_count > PATCHBLOB_TARGET_V2_MAX_TENSORS ||
      target->invocation_index >= invocation_count ||
      target->range_index >= ranges->size / ACT_RANGE_SIZE) {
    return unsupported(diagnostic, "version 2 target is outside the supported ACT contract");
  }
  input_count = target->tensor_count - 1u;
  output = &target->tensors[input_count];
  for (index = 0; index < target->tensor_count; ++index) {
    const patchblob_tensor_contract_v2 *tensor = &target->tensors[index];
    uint32_t dtype = dtype_from_public_precision_v2(tensor->precision);
    uint32_t expected_role =
        index < input_count ? PATCHBLOB_TENSOR_ROLE_INPUT_V2 : PATCHBLOB_TENSOR_ROLE_OUTPUT_V2;
    uint32_t expected_index = index < input_count ? (uint32_t)index : 0u;
    uint32_t expected_flags = index < input_count ? input_flags : output_flags;
    if (tensor->struct_size < sizeof(*tensor) || tensor->role != expected_role ||
        tensor->tensor_index != expected_index || dtype == 0 || tensor->element_count == 0 ||
        tensor->span_bytes == 0 || tensor->observed_flags != expected_flags) {
      return unsupported(diagnostic,
                         "version 2 tensor records must use the observed static dense CMX layout");
    }
    if (tensor->element_count > UINT64_MAX / dtype_table[dtype].size_bytes ||
        tensor->span_bytes != tensor->element_count * dtype_table[dtype].size_bytes) {
      return unsupported(diagnostic, "version 2 tensor byte span does not match its precision");
    }
    if (index > 0 && tensor->precision != target->tensors[0].precision) {
      mixed_precision = true;
    }
  }
  if (mixed_precision) {
    supported_unary_conversion =
        (target->tensors[0].precision == PATCHBLOB_TENSOR_PRECISION_FP16_V2 &&
         output->precision == PATCHBLOB_TENSOR_PRECISION_FP32_V2) ||
        (target->tensors[0].precision == PATCHBLOB_TENSOR_PRECISION_FP32_V2 &&
         output->precision == PATCHBLOB_TENSOR_PRECISION_FP16_V2) ||
        (target->tensors[0].precision == PATCHBLOB_TENSOR_PRECISION_U8_V2 &&
         output->precision == PATCHBLOB_TENSOR_PRECISION_FP32_V2);
    if (target->target_flags != PATCHBLOB_TARGET_UNARY_CONVERSION_V2 || input_count != 1u ||
        target->tensors[0].precision == output->precision ||
        target->tensors[0].element_count != output->element_count || !supported_unary_conversion) {
      return unsupported(diagnostic,
                         "mixed precision is limited to the observed unary FP16/FP32 or "
                         "U8-to-FP32 conversion contracts");
    }
  } else {
    if (target->target_flags != 0u) {
      return unsupported(diagnostic, "version 2 same-precision tensor contract is unsupported");
    }
    for (index = 0; index < input_count; ++index) {
      const patchblob_tensor_contract_v2 *input = &target->tensors[index];
      bool observed_scalar = output->element_count > 1u && input->element_count == 1u;
      if (observed_scalar) {
        ++scalar_input_count;
      } else if (input->element_count != output->element_count ||
                 input->span_bytes != output->span_bytes) {
        return unsupported(diagnostic, "version 2 input tensor shape is unsupported");
      }
    }
    supported_multi_scalar_fp16 =
        output->precision == PATCHBLOB_TENSOR_PRECISION_FP16_V2 && input_count == 5u &&
        ((output->element_count == 16u && output->span_bytes == 32u) ||
         (output->element_count == 8u && output->span_bytes == 16u)) &&
        target->tensors[0].element_count == output->element_count &&
        target->tensors[0].span_bytes == output->span_bytes && scalar_input_count == 4u;
    supported_hybrid_scalar_fp16 =
        output->precision == PATCHBLOB_TENSOR_PRECISION_FP16_V2 && input_count == 5u &&
        ((output->element_count == 4u && output->span_bytes == 8u) ||
         (output->element_count == 16u && output->span_bytes == 32u)) &&
        target->tensors[0].element_count == output->element_count &&
        target->tensors[0].span_bytes == output->span_bytes &&
        target->tensors[1].element_count == output->element_count &&
        target->tensors[1].span_bytes == output->span_bytes && scalar_input_count == 3u;
    if (input_count == 2u && scalar_input_count > 1u) {
      return unsupported(diagnostic,
                         "version 2 binary contracts allow at most one observed scalar input");
    }
    if (scalar_input_count != 0u && !(input_count == 2u && scalar_input_count == 1u) &&
        !supported_multi_scalar_fp16 && !supported_hybrid_scalar_fp16) {
      return unsupported(diagnostic,
                         "version 2 scalar inputs do not match an observed binary or FP16 "
                         "five-input contract");
    }
    if ((output->precision == PATCHBLOB_TENSOR_PRECISION_I32_V2 ||
         output->precision == PATCHBLOB_TENSOR_PRECISION_I16_V2 ||
         output->precision == PATCHBLOB_TENSOR_PRECISION_I8_V2) &&
        (input_count > 2u || scalar_input_count != 0u)) {
      return unsupported(diagnostic,
                         "signed integer custom kernels are limited to observed unary or "
                         "equal-span binary contracts");
    }
    if (output->precision == PATCHBLOB_TENSOR_PRECISION_U8_V2) {
      return unsupported(diagnostic,
                         "U8 custom kernels are limited to the observed U8-to-FP32 conversion");
    }
    if (output->precision == PATCHBLOB_TENSOR_PRECISION_U16_V2 && input_count != 2u) {
      return unsupported(
          diagnostic,
          "U16 custom kernels are limited to observed equal-span or one-scalar binary contracts");
    }
    if (output->precision == PATCHBLOB_TENSOR_PRECISION_U32_V2 && input_count != 2u) {
      return unsupported(
          diagnostic,
          "U32 custom kernels are limited to observed equal-span or one-scalar binary contracts");
    }
  }

  invocation_offset = invocations->offset + (size_t)target->invocation_index * ACT_INVOCATION_SIZE;
  if (read_u32(layout->blob.data + invocation_offset) != target->range_index) {
    return unsupported(diagnostic, "selected invocation does not reference the selected range");
  }
  base = parameter_bases[target->invocation_index];
  limit = invocation_parameter_limit(parameter_bases, invocation_count, base, params->size);
  if (limit - base < (uint64_t)target->tensor_count * MEMREF_SIZE) {
    return unsupported(diagnostic, "selected invocation parameter block is too small");
  }
  {
    size_t special_count;
    npunlock_status status =
        count_data_pointer_relocations(layout, base, limit, &special_count, diagnostic);
    if (status != NPUNLOCK_STATUS_OK) {
      return status;
    }
    if (special_count != target->tensor_count) {
      return unsupported(
          diagnostic, "selected invocation parameter block has an unexpected data-pointer layout");
    }
  }
  for (index = 0; index < target->tensor_count; ++index) {
    const patchblob_tensor_contract_v2 *expected = &target->tensors[index];
    uint32_t dtype = dtype_from_public_precision_v2(expected->precision);
    npunlock_status status =
        validate_memref(layout, base + (uint64_t)index * MEMREF_SIZE, &actual[index], diagnostic);
    if (status != NPUNLOCK_STATUS_OK) {
      return status;
    }
    if (actual[index].element_count != expected->element_count ||
        actual[index].span_bytes != expected->span_bytes || actual[index].dtype_kind != dtype) {
      return unsupported(diagnostic, "tensor does not match the version 2 element contract");
    }
  }
  for (index = 0; index < input_count; ++index) {
    if (actual[index].data_symbol != actual[input_count].data_symbol) {
      return unsupported(diagnostic,
                         "selected invocation tensors do not share the observed address space");
    }
    if (spans_overlap((uint64_t)actual[index].data_addend, actual[index].span_bytes,
                      (uint64_t)actual[input_count].data_addend, actual[input_count].span_bytes)) {
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
  detail->input_count = (uint32_t)input_count;
  detail->parameter_base = base;
  detail->element_count = output->element_count;
  detail->span_bytes = output->span_bytes;
  detail->target_flags_v2 = target->target_flags;
  detail->tensor_count_v2 = target->tensor_count;
  memcpy(detail->tensors_v2, target->tensors,
         (size_t)target->tensor_count * sizeof(target->tensors[0]));
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
        candidate.type == R_VPU_32 && !candidate.special_source &&
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
    memref_contract second_input;
    memref_contract output;
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
    if (target->expected_input_count == 2u) {
      status = validate_memref(&layout, base + MEMREF_SIZE, &second_input, diagnostic);
      if (status != NPUNLOCK_STATUS_OK) {
        goto fail;
      }
      status = validate_memref(&layout, base + 2u * MEMREF_SIZE, &output, diagnostic);
      if (status != NPUNLOCK_STATUS_OK) {
        goto fail;
      }
      if (first_input.element_count > 1u && second_input.dtype_kind == first_input.dtype_kind &&
          second_input.element_count == 1u &&
          second_input.span_bytes == dtype_table[first_input.dtype_kind].size_bytes &&
          output.dtype_kind == first_input.dtype_kind &&
          output.element_count == first_input.element_count &&
          output.span_bytes == first_input.span_bytes) {
        target->required_contract_flags |= PATCHBLOB_CONTRACT_INPUT_1_SCALAR;
      }
    }
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

static bool tensor_contracts_v2_equal(const patchblob_target_v2 *left,
                                      const patchblob_target_v2 *right) {
  size_t index;
  if (left->tensor_count != right->tensor_count || left->target_flags != right->target_flags) {
    return false;
  }
  for (index = 0; index < left->tensor_count; ++index) {
    const patchblob_tensor_contract_v2 *a = &left->tensors[index];
    const patchblob_tensor_contract_v2 *b = &right->tensors[index];
    if (a->role != b->role || a->tensor_index != b->tensor_index || a->precision != b->precision ||
        a->element_count != b->element_count || a->span_bytes != b->span_bytes ||
        a->observed_flags != b->observed_flags) {
      return false;
    }
  }
  return true;
}

npunlock_status npunlock_discover_graph_targets_v2(npunlock_view graph_blob,
                                                   patchblob_discovered_target_v2 **targets,
                                                   size_t *target_count, size_t *group_count,
                                                   npunlock_diagnostic *diagnostic) {
  graph_layout layout;
  uint64_t *parameter_bases = NULL;
  patchblob_discovered_target_v2 *discovered = NULL;
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
  discovered = (patchblob_discovered_target_v2 *)calloc(invocation_count, sizeof(*discovered));
  if (discovered == NULL) {
    status = npunlock_set_diagnostic(diagnostic, NPUNLOCK_STATUS_OUT_OF_MEMORY,
                                     "patchblob.discover_v2", "could not allocate target metadata");
    goto fail;
  }
  for (index = 0; index < invocation_count; ++index) {
    const graph_section *invocations = &layout.sections[layout.invocations_index];
    const graph_section *params = &layout.sections[layout.params_index];
    const uint8_t *record = layout.blob.data + invocations->offset + index * ACT_INVOCATION_SIZE;
    patchblob_target_v2 *target = &discovered[index].target;
    npunlock_patch_detail detail;
    graph_relocation range_relocation;
    uint64_t base = parameter_bases[index];
    uint64_t limit =
        invocation_parameter_limit(parameter_bases, invocation_count, base, params->size);
    size_t record_count;
    size_t tensor_index;
    size_t other;

    status = count_data_pointer_relocations(&layout, base, limit, &record_count, diagnostic);
    if (status != NPUNLOCK_STATUS_OK) {
      goto fail;
    }
    if (record_count < 2u || record_count > PATCHBLOB_TARGET_V2_MAX_TENSORS ||
        limit - base < record_count * MEMREF_SIZE) {
      status = unsupported(diagnostic,
                           "ACT invocation does not have a discoverable input/output contract");
      goto fail;
    }
    discovered[index].struct_size = (uint32_t)sizeof(discovered[index]);
    target->struct_size = (uint32_t)sizeof(*target);
    target->abi_version = PATCHBLOB_TARGET_V2_ABI_VERSION;
    target->invocation_index = (uint32_t)index;
    target->range_index = read_u32(record);
    target->tensor_count = (uint32_t)record_count;
    for (tensor_index = 0; tensor_index < record_count; ++tensor_index) {
      memref_contract actual;
      patchblob_tensor_contract_v2 *tensor = &target->tensors[tensor_index];
      bool is_output = tensor_index + 1u == record_count;
      status = validate_memref(&layout, base + (uint64_t)tensor_index * MEMREF_SIZE, &actual,
                               diagnostic);
      if (status != NPUNLOCK_STATUS_OK) {
        goto fail;
      }
      tensor->struct_size = (uint32_t)sizeof(*tensor);
      tensor->role = is_output ? PATCHBLOB_TENSOR_ROLE_OUTPUT_V2 : PATCHBLOB_TENSOR_ROLE_INPUT_V2;
      tensor->tensor_index = is_output ? 0u : (uint32_t)tensor_index;
      tensor->precision = public_precision_v2(actual.dtype_kind);
      tensor->element_count = actual.element_count;
      tensor->span_bytes = actual.span_bytes;
      tensor->observed_flags = PATCHBLOB_TENSOR_STATIC_V2 | PATCHBLOB_TENSOR_DENSE_V2 |
                               PATCHBLOB_TENSOR_CMX_V2 |
                               (is_output ? PATCHBLOB_TENSOR_DISJOINT_FROM_INPUTS_V2 : 0u);
    }
    for (tensor_index = 1; tensor_index < record_count; ++tensor_index) {
      if (target->tensors[tensor_index].precision != target->tensors[0].precision) {
        target->target_flags = PATCHBLOB_TARGET_UNARY_CONVERSION_V2;
      }
    }
    status =
        validate_target_v2(&layout, parameter_bases, invocation_count, target, &detail, diagnostic);
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
    } else if (!tensor_contracts_v2_equal(&discovered[index - 1u].target, target)) {
      status = unsupported(diagnostic, "ACT group contains inconsistent tensor contracts");
      goto fail;
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

static bool name_matches_suffix(const char *name, const char *prefix, const char *suffix) {
  size_t name_size = strlen(name);
  size_t prefix_size = strlen(prefix);
  size_t suffix_size = strlen(suffix);
  return name_size == prefix_size + suffix_size && memcmp(name, prefix, prefix_size) == 0 &&
         memcmp(name + prefix_size, suffix, suffix_size) == 0;
}

static npunlock_status
inspect_network_output_routes(const graph_layout *layout,
                              npunlock_network_output_route_observation **routes_out,
                              size_t *route_count_out, npunlock_diagnostic *diagnostic) {
  npunlock_network_output_route_observation *routes = NULL;
  size_t route_count = 0;
  size_t route_index = 0;
  size_t section_index;

  *routes_out = NULL;
  *route_count_out = 0;
  for (section_index = 0; section_index < layout->section_count; ++section_index) {
    const graph_section *section = &layout->sections[section_index];
    size_t entries;
    if (!name_has_decimal_suffix(section->name, ".rlt.DMA_NetOutput")) {
      continue;
    }
    if (section->type != SHT_RELA || section->flags != 0x500040u || section->address != 0u ||
        section->alignment != 8u || section->entry_size != ELF64_RELA_SIZE ||
        section->size % ELF64_RELA_SIZE != 0u || section->info >= layout->section_count ||
        section->link >= layout->section_count) {
      return unsupported(diagnostic, "network-output relocation section is unsupported");
    }
    entries = section->size / ELF64_RELA_SIZE;
    if (entries == 0u) {
      return unsupported(diagnostic, "network-output relocation section is empty");
    }
    if (!npunlock_checked_add_size(route_count, entries, &route_count)) {
      return malformed(diagnostic, "network-output route count overflows");
    }
  }
  if (route_count == 0u) {
    return unsupported(diagnostic, "graph has no inspectable network-output DMA route");
  }
  if (route_count > SIZE_MAX / sizeof(*routes)) {
    return malformed(diagnostic, "network-output route allocation size overflows");
  }
  routes = (npunlock_network_output_route_observation *)calloc(route_count, sizeof(*routes));
  if (routes == NULL) {
    return npunlock_set_diagnostic(diagnostic, NPUNLOCK_STATUS_OUT_OF_MEMORY,
                                   "patchblob.inspect_routing",
                                   "could not allocate network-output route observations");
  }

  for (section_index = 0; section_index < layout->section_count; ++section_index) {
    const graph_section *section = &layout->sections[section_index];
    const graph_section *target;
    const graph_section *symbols;
    const graph_section *strings;
    const graph_section *runtime_relocations = NULL;
    const char *suffix;
    size_t candidate_index;
    size_t runtime_section_count = 0;
    size_t entry_index;
    if (!name_has_decimal_suffix(section->name, ".rlt.DMA_NetOutput")) {
      continue;
    }
    suffix = section->name + strlen(".rlt.DMA_NetOutput");
    target = &layout->sections[section->info];
    symbols = &layout->sections[section->link];
    if (!name_matches_suffix(target->name, ".text.dmaTasks", suffix) ||
        target->type != SHT_PROGBITS || target->alignment != 64u || target->entry_size != 1u ||
        target->size == 0u || target->size % DMA_TASK_SIZE != 0u ||
        target->size / DMA_TASK_SIZE > UINT32_MAX) {
      free(routes);
      return unsupported(diagnostic,
                         "network-output relocation targets an unsupported DMA section");
    }
    if (strcmp(symbols->name, ".symtab.output") != 0 || symbols->type != SHT_SYMTAB ||
        symbols->flags != 0x400000u || symbols->address != 0u || symbols->info != 0u ||
        symbols->alignment != 8u || symbols->entry_size != ELF64_SYMBOL_SIZE ||
        symbols->size % ELF64_SYMBOL_SIZE != 0u || symbols->link >= layout->section_count) {
      free(routes);
      return unsupported(diagnostic, "network-output relocation symbol table is unsupported");
    }
    strings = &layout->sections[symbols->link];
    if (strcmp(strings->name, ".symstrtab") != 0 || strings->type != SHT_STRTAB ||
        strings->flags != 0u || strings->address != 0u || strings->link != 0u ||
        strings->info != 0u || strings->alignment != 4u || strings->entry_size != 0u ||
        strings->size == 0u || layout->blob.data[strings->offset] != 0u) {
      free(routes);
      return unsupported(diagnostic, "network-output symbol strings are unsupported");
    }
    for (candidate_index = 0; candidate_index < layout->section_count; ++candidate_index) {
      const graph_section *candidate = &layout->sections[candidate_index];
      if (name_matches_suffix(candidate->name, ".rlt.text.dmaTasks", suffix) &&
          candidate->info == section->info && candidate->link == VPU_RT_SYMTAB) {
        runtime_relocations = candidate;
        ++runtime_section_count;
      }
    }
    if (runtime_section_count != 1u || runtime_relocations == NULL ||
        runtime_relocations->type != SHT_RELA || runtime_relocations->flags != 0x40u ||
        runtime_relocations->address != 0u || runtime_relocations->link != VPU_RT_SYMTAB ||
        runtime_relocations->alignment != 8u ||
        runtime_relocations->entry_size != ELF64_RELA_SIZE ||
        runtime_relocations->size % ELF64_RELA_SIZE != 0u) {
      free(routes);
      return unsupported(diagnostic, "network-output DMA runtime relocations are unsupported");
    }

    for (entry_index = 0; entry_index < section->size / ELF64_RELA_SIZE; ++entry_index) {
      const uint8_t *entry = layout->blob.data + section->offset + entry_index * ELF64_RELA_SIZE;
      uint64_t target_offset = read_u64(entry);
      uint64_t info = read_u64(entry + 8u);
      uint32_t symbol_index = (uint32_t)(info >> 32);
      uint32_t relocation_type = (uint32_t)info;
      int64_t output_addend = read_i64(entry + 16u);
      const uint8_t *symbol;
      uint32_t symbol_name_offset;
      uint64_t output_size;
      size_t task_base;
      uint32_t length;
      int64_t cmx_addend = 0;
      uint32_t cmx_symbol = 0;
      size_t source_matches = 0;
      size_t source_index;

      if (relocation_type != R_VPU_64 || symbol_index == 0u ||
          symbol_index >= symbols->size / ELF64_SYMBOL_SIZE || output_addend < 0 ||
          target_offset > SIZE_MAX || target_offset % DMA_TASK_SIZE != 0x18u ||
          !range_in_bounds((size_t)target_offset, 8u, target->size)) {
        free(routes);
        return unsupported(diagnostic, "network-output DMA destination relocation is unsupported");
      }
      symbol = layout->blob.data + symbols->offset + (size_t)symbol_index * ELF64_SYMBOL_SIZE;
      symbol_name_offset = read_u32(symbol);
      output_size = read_u64(symbol + 16u);
      if (symbol[4] != 0u || symbol[5] != 0u || read_u16(symbol + 6u) != 0u ||
          read_u64(symbol + 8u) != 0u || output_size == 0u || symbol_name_offset >= strings->size ||
          memchr(layout->blob.data + strings->offset + symbol_name_offset, 0,
                 strings->size - symbol_name_offset) == NULL) {
        free(routes);
        return unsupported(diagnostic, "network-output symbol is unsupported");
      }
      task_base = (size_t)target_offset - 0x18u;
      length = read_u32(layout->blob.data + target->offset + task_base + 0x20u);
      if (length == 0u || (uint64_t)output_addend > output_size ||
          (uint64_t)length > output_size - (uint64_t)output_addend) {
        free(routes);
        return unsupported(diagnostic, "network-output DMA length exceeds its output tensor");
      }
      for (source_index = 0; source_index < runtime_relocations->size / ELF64_RELA_SIZE;
           ++source_index) {
        const uint8_t *source_entry =
            layout->blob.data + runtime_relocations->offset + source_index * ELF64_RELA_SIZE;
        uint64_t source_offset = read_u64(source_entry);
        uint64_t source_info = read_u64(source_entry + 8u);
        if (source_offset == target_offset - 8u && (uint32_t)source_info == R_VPU_64 &&
            (uint32_t)(source_info >> 32) == VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR) {
          cmx_addend = read_i64(source_entry + 16u);
          cmx_symbol = (uint32_t)(source_info >> 32);
          ++source_matches;
        }
      }
      if (source_matches != 1u || cmx_addend < 0 ||
          (uint64_t)cmx_addend > UINT64_MAX - (uint64_t)length) {
        free(routes);
        return unsupported(diagnostic, "network-output DMA CMX source is missing or unsupported");
      }
      routes[route_index].dma_section_index = (uint32_t)section->info;
      routes[route_index].dma_task_index = (uint32_t)(task_base / DMA_TASK_SIZE);
      routes[route_index].cmx_source_symbol_raw = cmx_symbol;
      routes[route_index].cmx_source_addend_raw = cmx_addend;
      routes[route_index].output_symbol_raw = symbol_index;
      routes[route_index].output_addend_raw = output_addend;
      routes[route_index].length_bytes = length;
      ++route_index;
    }
  }
  if (route_index != route_count) {
    free(routes);
    return malformed(diagnostic, "network-output route count changed during inspection");
  }
  *routes_out = routes;
  *route_count_out = route_count;
  return NPUNLOCK_STATUS_OK;
}

static uint32_t dpu_input_element_size(uint32_t dtype) {
  switch (dtype) {
  case 0u: /* FP16 */
  case 5u: /* BF16 */
    return 2u;
  case 1u: /* U8 */
  case 2u: /* I8 */
  case 8u: /* FP8 */
    return 1u;
  default:
    return 0u;
  }
}

static uint32_t dpu_output_element_size(uint32_t dtype) {
  switch (dtype) {
  case 0u: /* FP16/BF16 */
    return 2u;
  case 1u: /* U8F */
  case 2u: /* G8 */
  case 3u: /* I8 */
    return 1u;
  case 4u: /* I32/FP32 */
    return 4u;
  default:
    return 0u;
  }
}

static bool dpu_dense_tensor_span(const uint32_t dimensions[3], uint32_t element_size,
                                  uint64_t *element_count, uint64_t *span_bytes) {
  uint64_t count = 1u;
  size_t index;
  if (element_size == 0u) {
    return false;
  }
  for (index = 0; index < 3u; ++index) {
    if (dimensions[index] == 0u || count > UINT64_MAX / dimensions[index]) {
      return false;
    }
    count *= dimensions[index];
  }
  if (count > UINT64_MAX / element_size) {
    return false;
  }
  *element_count = count;
  *span_bytes = count * element_size;
  return true;
}

static npunlock_status inspect_dpu_invariants(const graph_layout *layout,
                                              npunlock_dpu_invariant_observation **observations_out,
                                              size_t *observation_count_out,
                                              npunlock_diagnostic *diagnostic) {
  const graph_section *invariants = NULL;
  const graph_section *relocations = NULL;
  npunlock_dpu_invariant_observation *observations = NULL;
  uint8_t *input_fields_seen = NULL;
  uint8_t *output_fields_seen = NULL;
  size_t invariant_section_count = 0;
  size_t relocation_section_count = 0;
  size_t invariant_section_index = SIZE_MAX;
  size_t invariant_count;
  size_t allocation_size;
  size_t section_index;
  size_t index;

  *observations_out = NULL;
  *observation_count_out = 0;
  for (section_index = 0; section_index < layout->section_count; ++section_index) {
    const graph_section *section = &layout->sections[section_index];
    if (strcmp(section->name, ".text.DPUInvariants") == 0) {
      invariants = section;
      invariant_section_index = section_index;
      ++invariant_section_count;
    } else if (strcmp(section->name, ".rlt.text.DPUInvariants") == 0) {
      relocations = section;
      ++relocation_section_count;
    }
  }
  if (invariant_section_count == 0u && relocation_section_count == 0u) {
    return NPUNLOCK_STATUS_OK;
  }
  if (invariant_section_count != 1u || relocation_section_count != 1u || invariants == NULL ||
      relocations == NULL) {
    return unsupported(diagnostic, "DPU invariant sections are missing or ambiguous");
  }
  if (invariants->type != SHT_PROGBITS || invariants->flags != 0x20000004u ||
      invariants->address != 0u || invariants->link != 0u || invariants->info != 0u ||
      invariants->alignment != 64u || invariants->entry_size != 1u || invariants->size == 0u ||
      invariants->size % DPU_INVARIANT_SIZE != 0u ||
      invariants->size / DPU_INVARIANT_SIZE > UINT32_MAX) {
    return unsupported(diagnostic, "DPU invariant section has an unsupported NPU3720 form");
  }
  if (relocations->type != SHT_RELA || relocations->flags != 0x40u || relocations->address != 0u ||
      relocations->link != VPU_RT_SYMTAB || relocations->info != invariant_section_index ||
      relocations->alignment != 8u || relocations->entry_size != ELF64_RELA_SIZE ||
      relocations->size == 0u || relocations->size % ELF64_RELA_SIZE != 0u) {
    return unsupported(diagnostic, "DPU invariant relocations have an unsupported NPU3720 form");
  }

  invariant_count = invariants->size / DPU_INVARIANT_SIZE;
  if (!npunlock_checked_mul_size(invariant_count, sizeof(*observations), &allocation_size)) {
    return malformed(diagnostic, "DPU invariant observation allocation overflows");
  }
  observations = (npunlock_dpu_invariant_observation *)calloc(1u, allocation_size);
  input_fields_seen = (uint8_t *)calloc(invariant_count, sizeof(*input_fields_seen));
  output_fields_seen = (uint8_t *)calloc(invariant_count, sizeof(*output_fields_seen));
  if (observations == NULL || input_fields_seen == NULL || output_fields_seen == NULL) {
    free(observations);
    free(input_fields_seen);
    free(output_fields_seen);
    return npunlock_set_diagnostic(diagnostic, NPUNLOCK_STATUS_OUT_OF_MEMORY,
                                   "patchblob.inspect_routing",
                                   "could not allocate DPU invariant observations");
  }
  for (index = 0; index < invariant_count; ++index) {
    const uint8_t *record = layout->blob.data + invariants->offset + index * DPU_INVARIANT_SIZE;
    npunlock_dpu_invariant_observation *observation = &observations[index];
    uint32_t tensor_size0 = read_u32(record + 0x40u);
    uint32_t tensor_size1 = read_u32(record + 0x44u);
    uint32_t tensor_mode = read_u32(record + 0x4cu);
    uint32_t kernel_pad_cfg = read_u32(record + 0x30u);
    uint32_t output_cfg = read_u32(record + 0x78u);
    uint32_t output_dim0 = read_u32(record + 0x88u);
    uint32_t output_dim1 = read_u32(record + 0x8cu);
    uint32_t input_element_size;
    uint32_t output_element_size;
    observation->section_index = (uint32_t)invariant_section_index;
    observation->invariant_index = (uint32_t)index;
    observation->wait_mask_raw = read_u64(record + 0x108u);
    observation->post_mask_raw = read_u64(record + 0x110u);
    observation->barrier_group_raw = record[0x118u];
    observation->barrier_mask_raw = record[0x119u];
    observation->start_after_raw = read_u32(record + 0x120u);
    observation->clean_after_raw = read_u32(record + 0x124u);
    observation->variant_count_raw = read_u16(record + 0x128u);
    observation->cluster_raw = record[0x12au];
    observation->input_dimensions_xyz[0] = tensor_size0 & 0x3fffu;
    observation->input_dimensions_xyz[1] = (tensor_size0 >> 14u) & 0x3fffu;
    observation->input_dimensions_xyz[2] = tensor_size1 & 0x3fffu;
    observation->input_dtype_raw = (tensor_mode >> 4u) & 0xfu;
    observation->input_dense_raw = (kernel_pad_cfg >> 15u) & 1u;
    observation->output_dimensions_xyz[0] = (output_dim1 & 0x1fffu) + 1u;
    observation->output_dimensions_xyz[1] = (output_dim0 & 0x1fffu) + 1u;
    observation->output_dimensions_xyz[2] = ((output_dim0 >> 13u) & 0x1fffu) + 1u;
    observation->output_dtype_raw = output_cfg & 0xfu;
    observation->output_sparse_raw = ((output_cfg >> 13u) & 1u) | ((output_cfg >> 15u) & 1u);
    input_element_size = dpu_input_element_size(observation->input_dtype_raw);
    output_element_size = dpu_output_element_size(observation->output_dtype_raw);
    if (observation->variant_count_raw == 0u || observation->cluster_raw >= layout->tile_count) {
      free(observations);
      free(input_fields_seen);
      free(output_fields_seen);
      return unsupported(diagnostic, "DPU invariant scheduling fields are unsupported");
    }
    if (observation->input_dense_raw != 1u || observation->output_sparse_raw != 0u ||
        ((output_cfg >> 17u) & 1u) != 1u ||
        !dpu_dense_tensor_span(observation->input_dimensions_xyz, input_element_size,
                               &observation->input_element_count, &observation->input_span_bytes) ||
        !dpu_dense_tensor_span(observation->output_dimensions_xyz, output_element_size,
                               &observation->output_element_count,
                               &observation->output_span_bytes)) {
      free(observations);
      free(input_fields_seen);
      free(output_fields_seen);
      return unsupported(diagnostic,
                         "DPU invariant tensor extent is not static dense and byte-addressable");
    }
  }

  for (index = 0; index < relocations->size / ELF64_RELA_SIZE; ++index) {
    const uint8_t *entry = layout->blob.data + relocations->offset + index * ELF64_RELA_SIZE;
    uint64_t target_offset_raw = read_u64(entry);
    uint64_t info = read_u64(entry + 8u);
    uint32_t relocation_type = (uint32_t)info;
    uint32_t symbol = (uint32_t)(info >> 32);
    int64_t addend = read_i64(entry + 16u);
    size_t target_offset;
    size_t write_size;
    size_t invariant_index;
    size_t field_offset;
    bool is_input;
    bool is_output;
    uint32_t field_index;
    uint8_t field_bit;
    npunlock_dpu_invariant_observation *observation;
    npunlock_dpu_cmx_address_observation *address;

    if (relocation_type == R_VPU_64_LSHIFT) {
      write_size = 8u;
    } else if (relocation_type == R_VPU_32 || relocation_type == R_VPU_32_MULTICAST_BASE) {
      write_size = 4u;
    } else {
      free(observations);
      free(input_fields_seen);
      free(output_fields_seen);
      return unsupported(diagnostic, "DPU invariant relocation type is unsupported");
    }
    if (symbol > VPU_NNRD_SYM_LAST || !u64_to_size(target_offset_raw, &target_offset) ||
        target_offset % write_size != 0u ||
        !range_in_bounds(target_offset, write_size, invariants->size)) {
      free(observations);
      free(input_fields_seen);
      free(output_fields_seen);
      return malformed(diagnostic, "DPU invariant relocation is outside its target");
    }
    invariant_index = target_offset / DPU_INVARIANT_SIZE;
    field_offset = target_offset % DPU_INVARIANT_SIZE;
    is_input = field_offset >= DPU_INPUT_ADDRESS_OFFSET &&
               field_offset < DPU_INPUT_ADDRESS_OFFSET + NPUNLOCK_DPU_MAX_INPUT_ADDRESSES * 4u &&
               (field_offset - DPU_INPUT_ADDRESS_OFFSET) % 4u == 0u;
    is_output = field_offset >= DPU_OUTPUT_ADDRESS_OFFSET &&
                field_offset < DPU_OUTPUT_ADDRESS_OFFSET + NPUNLOCK_DPU_MAX_OUTPUT_ADDRESSES * 4u &&
                (field_offset - DPU_OUTPUT_ADDRESS_OFFSET) % 4u == 0u;
    if (!is_input && !is_output) {
      continue;
    }
    if (symbol != VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR || addend < 0 ||
        (is_input && relocation_type != R_VPU_32) ||
        (is_output && relocation_type != R_VPU_32_MULTICAST_BASE)) {
      free(observations);
      free(input_fields_seen);
      free(output_fields_seen);
      return unsupported(diagnostic, "DPU CMX address relocation is unsupported");
    }
    field_index = (uint32_t)((field_offset -
                              (is_input ? DPU_INPUT_ADDRESS_OFFSET : DPU_OUTPUT_ADDRESS_OFFSET)) /
                             4u);
    field_bit = (uint8_t)(1u << field_index);
    observation = &observations[invariant_index];
    if (is_input) {
      if ((input_fields_seen[invariant_index] & field_bit) != 0u ||
          observation->input_count == NPUNLOCK_DPU_MAX_INPUT_ADDRESSES) {
        free(observations);
        free(input_fields_seen);
        free(output_fields_seen);
        return unsupported(diagnostic, "DPU input CMX relocation is duplicated");
      }
      input_fields_seen[invariant_index] |= field_bit;
      address = &observation->inputs[observation->input_count++];
    } else {
      if ((output_fields_seen[invariant_index] & field_bit) != 0u ||
          observation->output_count == NPUNLOCK_DPU_MAX_OUTPUT_ADDRESSES) {
        free(observations);
        free(input_fields_seen);
        free(output_fields_seen);
        return unsupported(diagnostic, "DPU output CMX relocation is duplicated");
      }
      output_fields_seen[invariant_index] |= field_bit;
      address = &observation->outputs[observation->output_count++];
    }
    address->field_index = field_index;
    address->relocation_type = relocation_type;
    address->data_symbol_raw = symbol;
    address->data_addend_raw = addend;
  }
  free(input_fields_seen);
  free(output_fields_seen);
  for (index = 0; index < invariant_count; ++index) {
    if (observations[index].input_count == 0u || observations[index].output_count == 0u) {
      free(observations);
      return unsupported(diagnostic, "DPU invariant has no inspectable input or output CMX route");
    }
  }
  *observations_out = observations;
  *observation_count_out = invariant_count;
  return NPUNLOCK_STATUS_OK;
}

static uint32_t act_schedule_hop_count(const npunlock_invocation_routing_observation *invocations,
                                       size_t producer_index, size_t consumer_index) {
  uint32_t barrier_hops[64];
  uint64_t mask;
  size_t index;

  if (producer_index >= consumer_index) {
    return 0;
  }
  for (index = 0; index < 64u; ++index) {
    barrier_hops[index] = UINT32_MAX;
  }
  mask = invocations[producer_index].post_mask_raw;
  for (index = 0; index < 64u; ++index) {
    if ((mask & (UINT64_C(1) << index)) != 0u) {
      barrier_hops[index] = 1u;
    }
  }
  for (index = producer_index + 1u; index <= consumer_index; ++index) {
    uint32_t minimum_hops = UINT32_MAX;
    size_t bit_index;
    mask = invocations[index].wait_mask_raw;
    for (bit_index = 0; bit_index < 64u; ++bit_index) {
      if ((mask & (UINT64_C(1) << bit_index)) != 0u && barrier_hops[bit_index] < minimum_hops) {
        minimum_hops = barrier_hops[bit_index];
      }
    }
    if (index == consumer_index) {
      return minimum_hops == UINT32_MAX ? 0u : minimum_hops;
    }
    if (minimum_hops == UINT32_MAX || minimum_hops == UINT32_MAX - 1u) {
      continue;
    }
    mask = invocations[index].post_mask_raw;
    for (bit_index = 0; bit_index < 64u; ++bit_index) {
      if ((mask & (UINT64_C(1) << bit_index)) != 0u &&
          minimum_hops + 1u < barrier_hops[bit_index]) {
        barrier_hops[bit_index] = minimum_hops + 1u;
      }
    }
  }
  return 0;
}

static bool act_edge_candidate_matches(const npunlock_invocation_routing_observation *invocations,
                                       size_t invocation_count, size_t producer_index,
                                       size_t consumer_index, size_t consumer_tensor_index,
                                       uint32_t *schedule_hops) {
  const npunlock_invocation_routing_observation *producer;
  const npunlock_invocation_routing_observation *consumer;
  size_t producer_tensor_index;
  const patchblob_tensor_contract_v2 *output;
  const patchblob_tensor_contract_v2 *input;
  const npunlock_tensor_routing_observation *output_routing;
  const npunlock_tensor_routing_observation *input_routing;
  uint32_t hops;

  if (producer_index >= invocation_count || consumer_index >= invocation_count ||
      schedule_hops == NULL) {
    return false;
  }
  producer = &invocations[producer_index];
  consumer = &invocations[consumer_index];
  producer_tensor_index = producer->target.tensor_count - 1u;
  output = &producer->target.tensors[producer_tensor_index];
  input = &consumer->target.tensors[consumer_tensor_index];
  output_routing = &producer->tensors[producer_tensor_index];
  input_routing = &consumer->tensors[consumer_tensor_index];
  hops = act_schedule_hop_count(invocations, producer_index, consumer_index);
  *schedule_hops = hops;

  return producer->target.invocation_index < consumer->target.invocation_index &&
         producer->group_index != consumer->group_index &&
         output->role == PATCHBLOB_TENSOR_ROLE_OUTPUT_V2 &&
         input->role == PATCHBLOB_TENSOR_ROLE_INPUT_V2 && output->precision == input->precision &&
         output->element_count == input->element_count && output->span_bytes == input->span_bytes &&
         output->span_bytes != 0u && output_routing->data_addend >= 0 &&
         (uint64_t)output_routing->data_addend <= UINT64_MAX - output->span_bytes &&
         output_routing->data_symbol == input_routing->data_symbol &&
         output_routing->data_addend == input_routing->data_addend && hops != 0u;
}

npunlock_status
npunlock_derive_act_edge_candidates(const npunlock_invocation_routing_observation *invocations,
                                    size_t invocation_count,
                                    npunlock_act_edge_candidate_observation **candidates,
                                    size_t *candidate_count, npunlock_diagnostic *diagnostic) {
  npunlock_act_edge_candidate_observation *result = NULL;
  size_t comparisons = 0;
  size_t result_count = 0;
  size_t allocation_size;
  size_t consumer_index;
  size_t result_index = 0;

  if (candidates == NULL || candidate_count == NULL || diagnostic == NULL ||
      (invocations == NULL && invocation_count != 0u)) {
    return NPUNLOCK_STATUS_INVALID_ARGUMENT;
  }
  *candidates = NULL;
  *candidate_count = 0;

  for (consumer_index = 0; consumer_index < invocation_count; ++consumer_index) {
    const npunlock_invocation_routing_observation *consumer = &invocations[consumer_index];
    size_t consumer_tensor_index;
    if (consumer->target.tensor_count < 2u ||
        consumer->target.tensor_count > PATCHBLOB_TARGET_V2_MAX_TENSORS) {
      return unsupported(diagnostic, "ACT edge candidate has an unsupported tensor count");
    }
    for (consumer_tensor_index = 0; consumer_tensor_index + 1u < consumer->target.tensor_count;
         ++consumer_tensor_index) {
      size_t producer_index;
      for (producer_index = 0; producer_index < invocation_count; ++producer_index) {
        const npunlock_invocation_routing_observation *producer = &invocations[producer_index];
        uint32_t schedule_hops;
        if (producer->target.tensor_count < 2u ||
            producer->target.tensor_count > PATCHBLOB_TARGET_V2_MAX_TENSORS) {
          return unsupported(diagnostic, "ACT edge candidate has an unsupported tensor count");
        }
        if (comparisons == ACT_EDGE_MAX_COMPARISONS) {
          return unsupported(diagnostic, "ACT edge candidate analysis exceeds its work bound");
        }
        ++comparisons;
        if (act_edge_candidate_matches(invocations, invocation_count, producer_index,
                                       consumer_index, consumer_tensor_index, &schedule_hops)) {
          if (result_count == SIZE_MAX) {
            return unsupported(diagnostic, "ACT edge candidate count overflows");
          }
          ++result_count;
        }
      }
    }
  }
  if (result_count == 0u) {
    return NPUNLOCK_STATUS_OK;
  }
  if (!npunlock_checked_mul_size(result_count, sizeof(*result), &allocation_size)) {
    return unsupported(diagnostic, "ACT edge candidate allocation overflows");
  }
  result = (npunlock_act_edge_candidate_observation *)calloc(1u, allocation_size);
  if (result == NULL) {
    return npunlock_set_diagnostic(diagnostic, NPUNLOCK_STATUS_OUT_OF_MEMORY,
                                   "patchblob.inspect_routing",
                                   "could not allocate ACT edge candidates");
  }

  for (consumer_index = 0; consumer_index < invocation_count; ++consumer_index) {
    const npunlock_invocation_routing_observation *consumer = &invocations[consumer_index];
    size_t consumer_tensor_index;
    for (consumer_tensor_index = 0; consumer_tensor_index + 1u < consumer->target.tensor_count;
         ++consumer_tensor_index) {
      size_t producer_index;
      size_t local_count = 0;
      for (producer_index = 0; producer_index < invocation_count; ++producer_index) {
        uint32_t schedule_hops;
        if (act_edge_candidate_matches(invocations, invocation_count, producer_index,
                                       consumer_index, consumer_tensor_index, &schedule_hops)) {
          ++local_count;
        }
      }
      if (local_count > UINT32_MAX) {
        free(result);
        return unsupported(diagnostic, "ACT edge ambiguity count exceeds the supported range");
      }
      for (producer_index = 0; producer_index < invocation_count; ++producer_index) {
        const npunlock_invocation_routing_observation *producer = &invocations[producer_index];
        size_t producer_tensor_index = producer->target.tensor_count - 1u;
        const npunlock_tensor_routing_observation *routing;
        npunlock_act_edge_candidate_observation *candidate;
        uint32_t schedule_hops;
        if (!act_edge_candidate_matches(invocations, invocation_count, producer_index,
                                        consumer_index, consumer_tensor_index, &schedule_hops)) {
          continue;
        }
        if (result_index >= result_count) {
          free(result);
          return unsupported(diagnostic, "ACT edge candidate count changed during inspection");
        }
        routing = &producer->tensors[producer_tensor_index];
        candidate = &result[result_index++];
        candidate->producer_group_index = producer->group_index;
        candidate->producer_invocation_index = producer->target.invocation_index;
        candidate->producer_tensor_record_index = (uint32_t)producer_tensor_index;
        candidate->consumer_group_index = consumer->group_index;
        candidate->consumer_invocation_index = consumer->target.invocation_index;
        candidate->consumer_tensor_record_index = (uint32_t)consumer_tensor_index;
        candidate->consumer_candidate_count = (uint32_t)local_count;
        candidate->schedule_hop_count = schedule_hops;
        candidate->data_symbol_raw = routing->data_symbol;
        candidate->data_addend_raw = routing->data_addend;
        candidate->span_bytes = producer->target.tensors[producer_tensor_index].span_bytes;
        candidate->barrier_intersection_raw = producer->post_mask_raw & consumer->wait_mask_raw;
      }
    }
  }
  if (result_index != result_count) {
    free(result);
    return unsupported(diagnostic, "ACT edge candidate count changed during inspection");
  }
  *candidates = result;
  *candidate_count = result_count;
  return NPUNLOCK_STATUS_OK;
}

static bool dpu_address_is_first(const npunlock_dpu_cmx_address_observation *addresses,
                                 size_t address_index) {
  size_t index;
  for (index = 0; index < address_index; ++index) {
    if (addresses[index].data_symbol_raw == addresses[address_index].data_symbol_raw &&
        addresses[index].data_addend_raw == addresses[address_index].data_addend_raw) {
      return false;
    }
  }
  return true;
}

static uint32_t dpu_address_field_mask(const npunlock_dpu_cmx_address_observation *addresses,
                                       size_t address_count, size_t address_index) {
  uint32_t mask = 0u;
  size_t index;
  for (index = 0; index < address_count; ++index) {
    if (addresses[index].data_symbol_raw == addresses[address_index].data_symbol_raw &&
        addresses[index].data_addend_raw == addresses[address_index].data_addend_raw) {
      mask |= UINT32_C(1) << addresses[index].field_index;
    }
  }
  return mask;
}

static bool act_dpu_candidate_matches(const npunlock_invocation_routing_observation *act,
                                      size_t tensor_index,
                                      const npunlock_dpu_invariant_observation *dpu,
                                      const npunlock_dpu_cmx_address_observation *dpu_address,
                                      uint32_t direction, uint64_t *barrier_intersection) {
  const npunlock_tensor_routing_observation *routing = &act->tensors[tensor_index];
  const patchblob_tensor_contract_v2 *tensor = &act->target.tensors[tensor_index];
  uint64_t act_start;
  uint64_t dpu_start;
  uint64_t dpu_span;
  uint64_t intersection;

  if (routing->data_addend < 0 || dpu_address->data_addend_raw < 0 || tensor->span_bytes == 0u) {
    return false;
  }
  act_start = (uint64_t)routing->data_addend;
  dpu_start = (uint64_t)dpu_address->data_addend_raw;
  dpu_span = direction == NPUNLOCK_ACT_DPU_DIRECTION_ACT_TO_DPU ? dpu->input_span_bytes
                                                                : dpu->output_span_bytes;
  intersection = direction == NPUNLOCK_ACT_DPU_DIRECTION_ACT_TO_DPU
                     ? act->post_mask_raw & dpu->wait_mask_raw
                     : dpu->post_mask_raw & act->wait_mask_raw;
  *barrier_intersection = intersection;
  return routing->data_symbol == dpu_address->data_symbol_raw && dpu_span != 0u &&
         intersection != 0u && act_start >= dpu_start &&
         act_start <= UINT64_MAX - tensor->span_bytes && dpu_start <= UINT64_MAX - dpu_span &&
         act_start + tensor->span_bytes <= dpu_start + dpu_span;
}

static npunlock_status derive_act_dpu_edge_candidates(
    const npunlock_invocation_routing_observation *invocations, size_t invocation_count,
    const npunlock_dpu_invariant_observation *dpu_invariants, size_t dpu_invariant_count,
    npunlock_act_dpu_edge_candidate_observation **candidates, size_t *candidate_count,
    npunlock_diagnostic *diagnostic) {
  npunlock_act_dpu_edge_candidate_observation *result = NULL;
  size_t comparisons = 0u;
  size_t result_count = 0u;
  size_t result_index = 0u;
  size_t allocation_size;
  size_t act_index;
  uint32_t direction;

  *candidates = NULL;
  *candidate_count = 0u;
  for (act_index = 0; act_index < invocation_count; ++act_index) {
    const npunlock_invocation_routing_observation *act = &invocations[act_index];
    for (direction = NPUNLOCK_ACT_DPU_DIRECTION_ACT_TO_DPU;
         direction <= NPUNLOCK_ACT_DPU_DIRECTION_DPU_TO_ACT; ++direction) {
      size_t tensor_begin =
          direction == NPUNLOCK_ACT_DPU_DIRECTION_ACT_TO_DPU ? act->target.tensor_count - 1u : 0u;
      size_t tensor_end = direction == NPUNLOCK_ACT_DPU_DIRECTION_ACT_TO_DPU
                              ? act->target.tensor_count
                              : act->target.tensor_count - 1u;
      size_t tensor_index;
      for (tensor_index = tensor_begin; tensor_index < tensor_end; ++tensor_index) {
        size_t dpu_index;
        for (dpu_index = 0; dpu_index < dpu_invariant_count; ++dpu_index) {
          const npunlock_dpu_invariant_observation *dpu = &dpu_invariants[dpu_index];
          const npunlock_dpu_cmx_address_observation *addresses =
              direction == NPUNLOCK_ACT_DPU_DIRECTION_ACT_TO_DPU ? dpu->inputs : dpu->outputs;
          size_t address_count = direction == NPUNLOCK_ACT_DPU_DIRECTION_ACT_TO_DPU
                                     ? dpu->input_count
                                     : dpu->output_count;
          size_t address_index;
          for (address_index = 0; address_index < address_count; ++address_index) {
            uint64_t barrier_intersection;
            if (!dpu_address_is_first(addresses, address_index)) {
              continue;
            }
            if (comparisons == ACT_EDGE_MAX_COMPARISONS) {
              return unsupported(diagnostic,
                                 "ACT/DPU edge candidate analysis exceeds its work bound");
            }
            ++comparisons;
            if (act_dpu_candidate_matches(act, tensor_index, dpu, &addresses[address_index],
                                          direction, &barrier_intersection)) {
              if (result_count == SIZE_MAX) {
                return unsupported(diagnostic, "ACT/DPU edge candidate count overflows");
              }
              ++result_count;
            }
          }
        }
      }
    }
  }
  if (result_count == 0u) {
    return NPUNLOCK_STATUS_OK;
  }
  if (!npunlock_checked_mul_size(result_count, sizeof(*result), &allocation_size)) {
    return unsupported(diagnostic, "ACT/DPU edge candidate allocation overflows");
  }
  result = (npunlock_act_dpu_edge_candidate_observation *)calloc(1u, allocation_size);
  if (result == NULL) {
    return npunlock_set_diagnostic(diagnostic, NPUNLOCK_STATUS_OUT_OF_MEMORY,
                                   "patchblob.inspect_routing",
                                   "could not allocate ACT/DPU edge candidates");
  }

  for (act_index = 0; act_index < invocation_count; ++act_index) {
    const npunlock_invocation_routing_observation *act = &invocations[act_index];
    for (direction = NPUNLOCK_ACT_DPU_DIRECTION_ACT_TO_DPU;
         direction <= NPUNLOCK_ACT_DPU_DIRECTION_DPU_TO_ACT; ++direction) {
      size_t tensor_begin =
          direction == NPUNLOCK_ACT_DPU_DIRECTION_ACT_TO_DPU ? act->target.tensor_count - 1u : 0u;
      size_t tensor_end = direction == NPUNLOCK_ACT_DPU_DIRECTION_ACT_TO_DPU
                              ? act->target.tensor_count
                              : act->target.tensor_count - 1u;
      size_t tensor_index;
      for (tensor_index = tensor_begin; tensor_index < tensor_end; ++tensor_index) {
        size_t local_count = 0u;
        size_t dpu_index;
        for (dpu_index = 0; dpu_index < dpu_invariant_count; ++dpu_index) {
          const npunlock_dpu_invariant_observation *dpu = &dpu_invariants[dpu_index];
          const npunlock_dpu_cmx_address_observation *addresses =
              direction == NPUNLOCK_ACT_DPU_DIRECTION_ACT_TO_DPU ? dpu->inputs : dpu->outputs;
          size_t address_count = direction == NPUNLOCK_ACT_DPU_DIRECTION_ACT_TO_DPU
                                     ? dpu->input_count
                                     : dpu->output_count;
          size_t address_index;
          for (address_index = 0; address_index < address_count; ++address_index) {
            uint64_t barrier_intersection;
            if (dpu_address_is_first(addresses, address_index) &&
                act_dpu_candidate_matches(act, tensor_index, dpu, &addresses[address_index],
                                          direction, &barrier_intersection)) {
              ++local_count;
            }
          }
        }
        if (local_count > UINT32_MAX) {
          free(result);
          return unsupported(diagnostic,
                             "ACT/DPU edge ambiguity count exceeds the supported range");
        }
        for (dpu_index = 0; dpu_index < dpu_invariant_count; ++dpu_index) {
          const npunlock_dpu_invariant_observation *dpu = &dpu_invariants[dpu_index];
          const npunlock_dpu_cmx_address_observation *addresses =
              direction == NPUNLOCK_ACT_DPU_DIRECTION_ACT_TO_DPU ? dpu->inputs : dpu->outputs;
          size_t address_count = direction == NPUNLOCK_ACT_DPU_DIRECTION_ACT_TO_DPU
                                     ? dpu->input_count
                                     : dpu->output_count;
          size_t address_index;
          for (address_index = 0; address_index < address_count; ++address_index) {
            const npunlock_dpu_cmx_address_observation *address = &addresses[address_index];
            npunlock_act_dpu_edge_candidate_observation *candidate;
            uint64_t barrier_intersection;
            if (!dpu_address_is_first(addresses, address_index) ||
                !act_dpu_candidate_matches(act, tensor_index, dpu, address, direction,
                                           &barrier_intersection)) {
              continue;
            }
            if (result_index >= result_count) {
              free(result);
              return unsupported(diagnostic,
                                 "ACT/DPU edge candidate count changed during inspection");
            }
            candidate = &result[result_index++];
            candidate->direction = direction;
            candidate->act_group_index = act->group_index;
            candidate->act_invocation_index = act->target.invocation_index;
            candidate->act_tensor_record_index = (uint32_t)tensor_index;
            candidate->dpu_section_index = dpu->section_index;
            candidate->dpu_invariant_index = dpu->invariant_index;
            candidate->dpu_field_mask_raw =
                dpu_address_field_mask(addresses, address_count, address_index);
            candidate->act_candidate_count = (uint32_t)local_count;
            candidate->data_symbol_raw = address->data_symbol_raw;
            candidate->act_data_addend_raw = act->tensors[tensor_index].data_addend;
            candidate->act_span_bytes = act->target.tensors[tensor_index].span_bytes;
            candidate->dpu_data_addend_raw = address->data_addend_raw;
            candidate->dpu_span_bytes = direction == NPUNLOCK_ACT_DPU_DIRECTION_ACT_TO_DPU
                                            ? dpu->input_span_bytes
                                            : dpu->output_span_bytes;
            candidate->barrier_intersection_raw = barrier_intersection;
          }
        }
      }
    }
  }
  if (result_index != result_count) {
    free(result);
    return unsupported(diagnostic, "ACT/DPU edge candidate count changed during inspection");
  }
  *candidates = result;
  *candidate_count = result_count;
  return NPUNLOCK_STATUS_OK;
}

static bool
act_output_slice_candidate_matches(const npunlock_invocation_routing_observation *act,
                                   const npunlock_network_output_route_observation *route,
                                   uint64_t *output_slice_addend_candidate) {
  size_t tensor_index;
  const npunlock_tensor_routing_observation *routing;
  const patchblob_tensor_contract_v2 *tensor;
  uint64_t act_start;
  uint64_t route_start;
  uint64_t route_span = route->length_bytes;
  uint64_t route_output_start;
  uint64_t relative_start;

  if (act->target.tensor_count == 0u) {
    return false;
  }
  tensor_index = act->target.tensor_count - 1u;
  routing = &act->tensors[tensor_index];
  tensor = &act->target.tensors[tensor_index];
  if (routing->data_addend < 0 || route->cmx_source_addend_raw < 0 ||
      route->output_addend_raw < 0 || tensor->span_bytes == 0u || route_span == 0u ||
      routing->data_symbol != route->cmx_source_symbol_raw) {
    return false;
  }
  act_start = (uint64_t)routing->data_addend;
  route_start = (uint64_t)route->cmx_source_addend_raw;
  route_output_start = (uint64_t)route->output_addend_raw;
  if (act_start < route_start || act_start > UINT64_MAX - tensor->span_bytes ||
      route_start > UINT64_MAX - route_span ||
      act_start + tensor->span_bytes > route_start + route_span) {
    return false;
  }
  relative_start = act_start - route_start;
  if (route_output_start > UINT64_MAX - relative_start ||
      route_output_start + relative_start > UINT64_MAX - tensor->span_bytes) {
    return false;
  }
  *output_slice_addend_candidate = route_output_start + relative_start;
  return true;
}

static npunlock_status derive_act_output_slice_candidates(
    const npunlock_invocation_routing_observation *invocations, size_t invocation_count,
    const npunlock_network_output_route_observation *routes, size_t route_count,
    npunlock_act_output_slice_candidate_observation **candidates, size_t *candidate_count,
    npunlock_diagnostic *diagnostic) {
  npunlock_act_output_slice_candidate_observation *result = NULL;
  size_t comparisons = 0u;
  size_t result_count = 0u;
  size_t result_index = 0u;
  size_t allocation_size;
  size_t act_index;

  *candidates = NULL;
  *candidate_count = 0u;
  for (act_index = 0; act_index < invocation_count; ++act_index) {
    size_t route_index;
    for (route_index = 0; route_index < route_count; ++route_index) {
      uint64_t output_slice_addend_candidate;
      if (comparisons == ACT_EDGE_MAX_COMPARISONS) {
        return unsupported(diagnostic,
                           "ACT output-slice candidate analysis exceeds its work bound");
      }
      ++comparisons;
      if (act_output_slice_candidate_matches(&invocations[act_index], &routes[route_index],
                                             &output_slice_addend_candidate)) {
        if (result_count == SIZE_MAX) {
          return unsupported(diagnostic, "ACT output-slice candidate count overflows");
        }
        ++result_count;
      }
    }
  }
  if (result_count == 0u) {
    return NPUNLOCK_STATUS_OK;
  }
  if (!npunlock_checked_mul_size(result_count, sizeof(*result), &allocation_size)) {
    return unsupported(diagnostic, "ACT output-slice candidate allocation overflows");
  }
  result = (npunlock_act_output_slice_candidate_observation *)calloc(1u, allocation_size);
  if (result == NULL) {
    return npunlock_set_diagnostic(diagnostic, NPUNLOCK_STATUS_OUT_OF_MEMORY,
                                   "patchblob.inspect_routing",
                                   "could not allocate ACT output-slice candidates");
  }

  for (act_index = 0; act_index < invocation_count; ++act_index) {
    const npunlock_invocation_routing_observation *act = &invocations[act_index];
    size_t tensor_index;
    size_t local_count = 0u;
    size_t route_index;
    if (act->target.tensor_count == 0u) {
      continue;
    }
    tensor_index = act->target.tensor_count - 1u;
    for (route_index = 0; route_index < route_count; ++route_index) {
      uint64_t output_slice_addend_candidate;
      if (act_output_slice_candidate_matches(act, &routes[route_index],
                                             &output_slice_addend_candidate)) {
        ++local_count;
      }
    }
    if (local_count > UINT32_MAX) {
      free(result);
      return unsupported(diagnostic,
                         "ACT output-slice route ambiguity exceeds the supported range");
    }
    for (route_index = 0; route_index < route_count; ++route_index) {
      const npunlock_network_output_route_observation *route = &routes[route_index];
      npunlock_act_output_slice_candidate_observation *candidate;
      uint64_t output_slice_addend_candidate;
      if (!act_output_slice_candidate_matches(act, route, &output_slice_addend_candidate)) {
        continue;
      }
      if (result_index >= result_count) {
        free(result);
        return unsupported(diagnostic,
                           "ACT output-slice candidate count changed during inspection");
      }
      candidate = &result[result_index++];
      candidate->act_group_index = act->group_index;
      candidate->act_invocation_index = act->target.invocation_index;
      candidate->act_tensor_record_index = (uint32_t)tensor_index;
      candidate->dma_section_index = route->dma_section_index;
      candidate->dma_task_index = route->dma_task_index;
      candidate->route_candidate_count = (uint32_t)local_count;
      candidate->cmx_source_symbol_raw = route->cmx_source_symbol_raw;
      candidate->act_data_addend_raw = act->tensors[tensor_index].data_addend;
      candidate->act_span_bytes = act->target.tensors[tensor_index].span_bytes;
      candidate->route_cmx_source_addend_raw = route->cmx_source_addend_raw;
      candidate->route_length_bytes = route->length_bytes;
      candidate->output_symbol_raw = route->output_symbol_raw;
      candidate->route_output_addend_raw = route->output_addend_raw;
      candidate->output_slice_addend_candidate = output_slice_addend_candidate;
    }
  }
  if (result_index != result_count) {
    free(result);
    return unsupported(diagnostic, "ACT output-slice candidate count changed during inspection");
  }
  *candidates = result;
  *candidate_count = result_count;
  return NPUNLOCK_STATUS_OK;
}

typedef struct output_byte_interval {
  uint64_t start;
  uint64_t end;
} output_byte_interval;

static int compare_output_byte_intervals(const void *left_pointer, const void *right_pointer) {
  const output_byte_interval *left = (const output_byte_interval *)left_pointer;
  const output_byte_interval *right = (const output_byte_interval *)right_pointer;
  if (left->start < right->start) {
    return -1;
  }
  if (left->start > right->start) {
    return 1;
  }
  if (left->end < right->end) {
    return -1;
  }
  return left->end > right->end ? 1 : 0;
}

static size_t merge_output_byte_intervals(output_byte_interval *intervals, size_t interval_count,
                                          bool *has_overlap) {
  size_t input_index;
  size_t merged_count = 0u;
  *has_overlap = false;
  if (interval_count == 0u) {
    return 0u;
  }
  qsort(intervals, interval_count, sizeof(*intervals), compare_output_byte_intervals);
  for (input_index = 0; input_index < interval_count; ++input_index) {
    output_byte_interval *previous;
    if (merged_count == 0u) {
      intervals[merged_count++] = intervals[input_index];
      continue;
    }
    previous = &intervals[merged_count - 1u];
    if (intervals[input_index].start < previous->end) {
      *has_overlap = true;
    }
    if (intervals[input_index].start <= previous->end) {
      if (intervals[input_index].end > previous->end) {
        previous->end = intervals[input_index].end;
      }
    } else {
      intervals[merged_count++] = intervals[input_index];
    }
  }
  return merged_count;
}

static bool
act_dpu_candidates_share_input(const npunlock_act_dpu_edge_candidate_observation *left,
                               const npunlock_act_dpu_edge_candidate_observation *right) {
  return left->direction == NPUNLOCK_ACT_DPU_DIRECTION_ACT_TO_DPU &&
         right->direction == NPUNLOCK_ACT_DPU_DIRECTION_ACT_TO_DPU &&
         left->dpu_section_index == right->dpu_section_index &&
         left->dpu_invariant_index == right->dpu_invariant_index &&
         left->data_symbol_raw == right->data_symbol_raw &&
         left->dpu_data_addend_raw == right->dpu_data_addend_raw &&
         left->dpu_span_bytes == right->dpu_span_bytes;
}

npunlock_status npunlock_derive_act_dpu_output_cover_observations(
    const npunlock_invocation_routing_observation *invocations, size_t invocation_count,
    size_t group_count, const npunlock_act_dpu_edge_candidate_observation *candidates,
    size_t candidate_count, npunlock_act_dpu_output_cover_observation **covers, size_t *cover_count,
    npunlock_diagnostic *diagnostic) {
  npunlock_act_dpu_output_cover_observation *result = NULL;
  output_byte_interval *intervals = NULL;
  size_t allocation_size;
  size_t work_bound;
  size_t work_term;
  size_t result_count = 0u;
  size_t group_index;

  if (covers == NULL || cover_count == NULL || diagnostic == NULL ||
      (invocations == NULL && invocation_count != 0u) ||
      (candidates == NULL && candidate_count != 0u)) {
    return NPUNLOCK_STATUS_INVALID_ARGUMENT;
  }
  *covers = NULL;
  *cover_count = 0u;
  if (group_count == 0u || candidate_count == 0u) {
    return NPUNLOCK_STATUS_OK;
  }
  if (invocation_count > UINT32_MAX || group_count > UINT32_MAX || candidate_count > UINT32_MAX ||
      !npunlock_checked_add_size(invocation_count, candidate_count, &work_term) ||
      !npunlock_checked_mul_size(candidate_count, work_term, &work_bound) ||
      !npunlock_checked_mul_size(group_count, work_bound, &work_bound) ||
      work_bound > ACT_EDGE_MAX_COMPARISONS ||
      !npunlock_checked_mul_size(group_count, sizeof(*result), &allocation_size)) {
    return unsupported(diagnostic, "ACT/DPU output-cover analysis exceeds its work bound");
  }
  result = (npunlock_act_dpu_output_cover_observation *)calloc(1u, allocation_size);
  if (!npunlock_checked_mul_size(candidate_count, sizeof(*intervals), &allocation_size)) {
    free(result);
    return unsupported(diagnostic, "ACT/DPU output-cover interval allocation overflows");
  }
  intervals = (output_byte_interval *)calloc(1u, allocation_size);
  if ((result == NULL && group_count != 0u) || (intervals == NULL && candidate_count != 0u)) {
    free(result);
    free(intervals);
    return npunlock_set_diagnostic(diagnostic, NPUNLOCK_STATUS_OUT_OF_MEMORY,
                                   "patchblob.inspect_routing",
                                   "could not allocate ACT/DPU output-cover observations");
  }
  for (group_index = 0; group_index < group_count; ++group_index) {
    npunlock_act_dpu_output_cover_observation *observation;
    bool participates = false;
    size_t invocation_index;
    size_t candidate_index;
    for (candidate_index = 0; candidate_index < candidate_count; ++candidate_index) {
      if (candidates[candidate_index].direction == NPUNLOCK_ACT_DPU_DIRECTION_ACT_TO_DPU &&
          candidates[candidate_index].act_group_index == group_index) {
        participates = true;
        break;
      }
    }
    if (!participates) {
      continue;
    }
    observation = &result[result_count++];
    observation->act_group_index = (uint32_t)group_index;
    for (invocation_index = 0; invocation_index < invocation_count; ++invocation_index) {
      const npunlock_act_dpu_edge_candidate_observation *unique = NULL;
      const npunlock_invocation_routing_observation *invocation = &invocations[invocation_index];
      size_t local_count = 0u;
      if (invocation->group_index != group_index) {
        continue;
      }
      ++observation->invocation_count;
      if (invocation->target.tensor_count == 0u) {
        ++observation->unresolved_invocation_count;
        observation->unresolved_flags |= NPUNLOCK_ACT_DPU_COVER_INVOCATION_UNCLASSIFIED;
        continue;
      }
      for (candidate_index = 0; candidate_index < candidate_count; ++candidate_index) {
        const npunlock_act_dpu_edge_candidate_observation *candidate = &candidates[candidate_index];
        if (candidate->direction == NPUNLOCK_ACT_DPU_DIRECTION_ACT_TO_DPU &&
            candidate->act_group_index == group_index &&
            candidate->act_invocation_index == invocation->target.invocation_index &&
            candidate->act_tensor_record_index == invocation->target.tensor_count - 1u) {
          unique = candidate;
          ++local_count;
        }
      }
      if (local_count == 1u && unique != NULL && unique->act_candidate_count == 1u) {
        ++observation->matched_invocation_count;
      } else {
        ++observation->unresolved_invocation_count;
        observation->unresolved_flags |= NPUNLOCK_ACT_DPU_COVER_INVOCATION_UNCLASSIFIED;
        if (local_count > 1u || (unique != NULL && unique->act_candidate_count != 1u)) {
          observation->unresolved_flags |= NPUNLOCK_ACT_DPU_COVER_CONSUMER_AMBIGUOUS;
        }
      }
    }
    for (candidate_index = 0; candidate_index < candidate_count; ++candidate_index) {
      const npunlock_act_dpu_edge_candidate_observation *endpoint = &candidates[candidate_index];
      size_t earlier_index;
      size_t interval_count = 0u;
      size_t merged_count;
      bool first = true;
      bool overlap;
      uint64_t endpoint_start;
      uint64_t endpoint_end;
      if (endpoint->direction != NPUNLOCK_ACT_DPU_DIRECTION_ACT_TO_DPU ||
          endpoint->act_group_index != group_index || endpoint->act_candidate_count != 1u) {
        continue;
      }
      for (earlier_index = 0; earlier_index < candidate_index; ++earlier_index) {
        if (candidates[earlier_index].act_group_index == group_index &&
            candidates[earlier_index].act_candidate_count == 1u &&
            act_dpu_candidates_share_input(&candidates[earlier_index], endpoint)) {
          first = false;
          break;
        }
      }
      if (!first) {
        continue;
      }
      ++observation->dpu_input_count;
      for (earlier_index = 0; earlier_index < candidate_count; ++earlier_index) {
        const npunlock_act_dpu_edge_candidate_observation *candidate = &candidates[earlier_index];
        uint64_t start;
        if (candidate->act_group_index != group_index || candidate->act_candidate_count != 1u ||
            !act_dpu_candidates_share_input(candidate, endpoint)) {
          continue;
        }
        if (candidate->act_data_addend_raw < 0 ||
            (uint64_t)candidate->act_data_addend_raw > UINT64_MAX - candidate->act_span_bytes) {
          observation->unresolved_flags |= NPUNLOCK_ACT_DPU_COVER_UNION_MISMATCH;
          continue;
        }
        start = (uint64_t)candidate->act_data_addend_raw;
        intervals[interval_count].start = start;
        intervals[interval_count].end = start + candidate->act_span_bytes;
        ++interval_count;
      }
      merged_count = merge_output_byte_intervals(intervals, interval_count, &overlap);
      if (overlap) {
        observation->unresolved_flags |= NPUNLOCK_ACT_DPU_COVER_SLICE_OVERLAP;
      }
      if (endpoint->dpu_data_addend_raw < 0 ||
          (uint64_t)endpoint->dpu_data_addend_raw > UINT64_MAX - endpoint->dpu_span_bytes) {
        observation->unresolved_flags |= NPUNLOCK_ACT_DPU_COVER_UNION_MISMATCH;
        continue;
      }
      endpoint_start = (uint64_t)endpoint->dpu_data_addend_raw;
      endpoint_end = endpoint_start + endpoint->dpu_span_bytes;
      if (!overlap && merged_count == 1u && intervals[0].start == endpoint_start &&
          intervals[0].end == endpoint_end) {
        ++observation->exact_cover_input_count;
      } else {
        observation->unresolved_flags |= NPUNLOCK_ACT_DPU_COVER_UNION_MISMATCH;
      }
    }
    if (observation->dpu_input_count == 0u) {
      observation->unresolved_flags |= NPUNLOCK_ACT_DPU_COVER_CONSUMER_AMBIGUOUS;
    }
    observation->exact_cover =
        observation->unresolved_flags == 0u && observation->invocation_count != 0u &&
        observation->matched_invocation_count == observation->invocation_count &&
        observation->exact_cover_input_count == observation->dpu_input_count;
  }
  free(intervals);
  *covers = result;
  *cover_count = result_count;
  return NPUNLOCK_STATUS_OK;
}

static bool
output_cover_symbol_is_first(const npunlock_act_output_slice_candidate_observation *candidates,
                             size_t candidate_index) {
  size_t index;
  for (index = 0; index < candidate_index; ++index) {
    if (candidates[index].output_symbol_raw == candidates[candidate_index].output_symbol_raw) {
      return false;
    }
  }
  return true;
}

npunlock_status npunlock_derive_act_output_cover_observations(
    const npunlock_invocation_routing_observation *invocations, size_t invocation_count,
    const npunlock_network_output_route_observation *routes, size_t route_count,
    const npunlock_act_output_slice_candidate_observation *slice_candidates,
    size_t slice_candidate_count, npunlock_act_output_cover_observation **covers,
    size_t *cover_count, npunlock_diagnostic *diagnostic) {
  npunlock_act_output_cover_observation *result = NULL;
  output_byte_interval *route_intervals = NULL;
  output_byte_interval *slice_intervals = NULL;
  size_t result_count = 0u;
  size_t result_index = 0u;
  size_t allocation_size;
  size_t candidate_index;
  size_t work_bound;
  size_t work_term;

  if (covers == NULL || cover_count == NULL || diagnostic == NULL ||
      (invocations == NULL && invocation_count != 0u) || (routes == NULL && route_count != 0u) ||
      (slice_candidates == NULL && slice_candidate_count != 0u)) {
    return NPUNLOCK_STATUS_INVALID_ARGUMENT;
  }
  *covers = NULL;
  *cover_count = 0u;
  if (!npunlock_checked_mul_size(slice_candidate_count, slice_candidate_count, &work_bound) ||
      work_bound > ACT_EDGE_MAX_COMPARISONS) {
    return unsupported(diagnostic, "ACT output-cover analysis exceeds its work bound");
  }
  for (candidate_index = 0; candidate_index < slice_candidate_count; ++candidate_index) {
    if (output_cover_symbol_is_first(slice_candidates, candidate_index)) {
      if (result_count == SIZE_MAX) {
        return unsupported(diagnostic, "ACT output-cover observation count overflows");
      }
      ++result_count;
    }
  }
  if (result_count == 0u) {
    return NPUNLOCK_STATUS_OK;
  }
  if (!npunlock_checked_mul_size(result_count, invocation_count, &work_term) ||
      !npunlock_checked_mul_size(work_term, slice_candidate_count, &work_term) ||
      !npunlock_checked_add_size(work_bound, work_term, &work_bound) ||
      !npunlock_checked_mul_size(result_count, route_count, &work_term) ||
      !npunlock_checked_add_size(work_bound, work_term, &work_bound) ||
      !npunlock_checked_mul_size(result_count, slice_candidate_count, &work_term) ||
      !npunlock_checked_add_size(work_bound, work_term, &work_bound) ||
      work_bound > ACT_EDGE_MAX_COMPARISONS) {
    return unsupported(diagnostic, "ACT output-cover analysis exceeds its work bound");
  }
  if (!npunlock_checked_mul_size(result_count, sizeof(*result), &allocation_size)) {
    return unsupported(diagnostic, "ACT output-cover observation allocation overflows");
  }
  result = (npunlock_act_output_cover_observation *)calloc(1u, allocation_size);
  if (result == NULL) {
    return npunlock_set_diagnostic(diagnostic, NPUNLOCK_STATUS_OUT_OF_MEMORY,
                                   "patchblob.inspect_routing",
                                   "could not allocate ACT output-cover observations");
  }
  if (!npunlock_checked_mul_size(route_count, sizeof(*route_intervals), &allocation_size) ||
      !npunlock_checked_mul_size(slice_candidate_count, sizeof(*slice_intervals),
                                 &allocation_size)) {
    free(result);
    return unsupported(diagnostic, "ACT output-cover interval allocation overflows");
  }
  route_intervals = (output_byte_interval *)calloc(route_count, sizeof(*route_intervals));
  slice_intervals = (output_byte_interval *)calloc(slice_candidate_count, sizeof(*slice_intervals));
  if ((route_intervals == NULL && route_count != 0u) ||
      (slice_intervals == NULL && slice_candidate_count != 0u)) {
    free(result);
    free(route_intervals);
    free(slice_intervals);
    return npunlock_set_diagnostic(diagnostic, NPUNLOCK_STATUS_OUT_OF_MEMORY,
                                   "patchblob.inspect_routing",
                                   "could not allocate ACT output-cover intervals");
  }

  for (candidate_index = 0; candidate_index < slice_candidate_count; ++candidate_index) {
    const npunlock_act_output_slice_candidate_observation *first =
        &slice_candidates[candidate_index];
    npunlock_act_output_cover_observation *observation;
    size_t act_groups = 0u;
    size_t group_invocations = 0u;
    size_t routed_invocations = 0u;
    size_t matching_routes = 0u;
    size_t matching_slices = 0u;
    size_t route_merged_count;
    size_t slice_merged_count;
    size_t index;
    bool route_overlap;
    bool slice_overlap;
    bool unions_equal = true;

    if (!output_cover_symbol_is_first(slice_candidates, candidate_index)) {
      continue;
    }
    if (result_index >= result_count) {
      free(result);
      free(route_intervals);
      free(slice_intervals);
      return unsupported(diagnostic,
                         "ACT output-cover observation count changed during inspection");
    }
    observation = &result[result_index++];
    observation->output_symbol_raw = first->output_symbol_raw;
    for (index = 0; index < slice_candidate_count; ++index) {
      size_t earlier_index;
      bool first_group = true;
      if (slice_candidates[index].output_symbol_raw != first->output_symbol_raw) {
        continue;
      }
      for (earlier_index = 0; earlier_index < index; ++earlier_index) {
        if (slice_candidates[earlier_index].output_symbol_raw == first->output_symbol_raw &&
            slice_candidates[earlier_index].act_group_index ==
                slice_candidates[index].act_group_index) {
          first_group = false;
          break;
        }
      }
      if (first_group) {
        ++act_groups;
      }
    }
    for (index = 0; index < invocation_count; ++index) {
      size_t inner_index;
      bool participating_group = false;
      bool routed = false;
      for (inner_index = 0; inner_index < slice_candidate_count; ++inner_index) {
        if (slice_candidates[inner_index].output_symbol_raw == first->output_symbol_raw &&
            slice_candidates[inner_index].act_group_index == invocations[index].group_index) {
          participating_group = true;
          break;
        }
      }
      if (!participating_group) {
        continue;
      }
      ++group_invocations;
      for (inner_index = 0; inner_index < slice_candidate_count; ++inner_index) {
        if (slice_candidates[inner_index].output_symbol_raw == first->output_symbol_raw &&
            slice_candidates[inner_index].act_invocation_index ==
                invocations[index].target.invocation_index) {
          routed = true;
          break;
        }
      }
      if (routed) {
        ++routed_invocations;
      }
    }
    for (index = 0; index < route_count; ++index) {
      uint64_t start;
      if (routes[index].output_symbol_raw != first->output_symbol_raw) {
        continue;
      }
      if (routes[index].output_addend_raw < 0) {
        observation->unresolved_flags |= NPUNLOCK_OUTPUT_COVER_UNION_MISMATCH;
        continue;
      }
      start = (uint64_t)routes[index].output_addend_raw;
      if (start > UINT64_MAX - routes[index].length_bytes) {
        observation->unresolved_flags |= NPUNLOCK_OUTPUT_COVER_UNION_MISMATCH;
        continue;
      }
      route_intervals[matching_routes].start = start;
      route_intervals[matching_routes].end = start + routes[index].length_bytes;
      if (observation->route_span_bytes > UINT64_MAX - routes[index].length_bytes) {
        free(result);
        free(route_intervals);
        free(slice_intervals);
        return unsupported(diagnostic, "ACT output-cover route span overflows");
      }
      observation->route_span_bytes += routes[index].length_bytes;
      ++matching_routes;
    }
    for (index = 0; index < slice_candidate_count; ++index) {
      const npunlock_act_output_slice_candidate_observation *candidate = &slice_candidates[index];
      uint64_t start;
      if (candidate->output_symbol_raw != first->output_symbol_raw) {
        continue;
      }
      if (candidate->route_candidate_count != 1u) {
        observation->unresolved_flags |= NPUNLOCK_OUTPUT_COVER_ROUTE_AMBIGUITY;
      }
      start = candidate->output_slice_addend_candidate;
      if (start > UINT64_MAX - candidate->act_span_bytes) {
        observation->unresolved_flags |= NPUNLOCK_OUTPUT_COVER_UNION_MISMATCH;
        continue;
      }
      slice_intervals[matching_slices].start = start;
      slice_intervals[matching_slices].end = start + candidate->act_span_bytes;
      if (observation->slice_span_bytes > UINT64_MAX - candidate->act_span_bytes) {
        free(result);
        free(route_intervals);
        free(slice_intervals);
        return unsupported(diagnostic, "ACT output-cover slice span overflows");
      }
      observation->slice_span_bytes += candidate->act_span_bytes;
      ++matching_slices;
    }
    route_merged_count =
        merge_output_byte_intervals(route_intervals, matching_routes, &route_overlap);
    slice_merged_count =
        merge_output_byte_intervals(slice_intervals, matching_slices, &slice_overlap);
    if (route_overlap) {
      observation->unresolved_flags |= NPUNLOCK_OUTPUT_COVER_ROUTE_OVERLAP;
    }
    if (slice_overlap) {
      observation->unresolved_flags |= NPUNLOCK_OUTPUT_COVER_SLICE_OVERLAP;
    }
    if (route_merged_count != slice_merged_count) {
      unions_equal = false;
    } else {
      for (index = 0; index < route_merged_count; ++index) {
        if (route_intervals[index].start != slice_intervals[index].start ||
            route_intervals[index].end != slice_intervals[index].end) {
          unions_equal = false;
          break;
        }
      }
    }
    if (!unions_equal) {
      observation->unresolved_flags |= NPUNLOCK_OUTPUT_COVER_UNION_MISMATCH;
    }
    if (act_groups > UINT32_MAX || group_invocations > UINT32_MAX ||
        routed_invocations > UINT32_MAX || matching_routes > UINT32_MAX ||
        matching_slices > UINT32_MAX) {
      free(result);
      free(route_intervals);
      free(slice_intervals);
      return unsupported(diagnostic, "ACT output-cover count exceeds the supported range");
    }
    observation->act_group_count = (uint32_t)act_groups;
    observation->group_invocation_count = (uint32_t)group_invocations;
    observation->routed_invocation_count = (uint32_t)routed_invocations;
    observation->unrouted_invocation_count = (uint32_t)(group_invocations - routed_invocations);
    observation->route_count = (uint32_t)matching_routes;
    observation->slice_candidate_count = (uint32_t)matching_slices;
    observation->exact_cover = observation->unresolved_flags == 0u;
  }
  free(route_intervals);
  free(slice_intervals);
  *covers = result;
  *cover_count = result_count;
  return NPUNLOCK_STATUS_OK;
}

static size_t invocation_slice_candidate_count(
    const npunlock_act_output_slice_candidate_observation *slice_candidates,
    size_t slice_candidate_count, uint32_t invocation_index,
    const npunlock_act_output_slice_candidate_observation **unique_candidate) {
  size_t count = 0u;
  size_t index;
  *unique_candidate = NULL;
  for (index = 0; index < slice_candidate_count; ++index) {
    if (slice_candidates[index].act_invocation_index == invocation_index) {
      *unique_candidate = &slice_candidates[index];
      ++count;
    }
  }
  if (count != 1u || (*unique_candidate)->route_candidate_count != 1u) {
    *unique_candidate = NULL;
  }
  return count;
}

static bool replica_contracts_match(const npunlock_invocation_routing_observation *left,
                                    const npunlock_invocation_routing_observation *right,
                                    int64_t *translation) {
  int64_t result = 0;
  size_t tensor_index;
  if (left->tile_candidate_raw == right->tile_candidate_raw ||
      left->wait_mask_raw != right->wait_mask_raw || left->post_mask_raw != right->post_mask_raw ||
      left->barrier_group_raw != right->barrier_group_raw ||
      left->barrier_mask_raw != right->barrier_mask_raw ||
      left->start_after_raw != right->start_after_raw ||
      left->clean_after_raw != right->clean_after_raw ||
      left->target.abi_version != right->target.abi_version ||
      left->target.tensor_count != right->target.tensor_count ||
      left->target.target_flags != right->target.target_flags) {
    return false;
  }
  for (tensor_index = 0; tensor_index < left->target.tensor_count; ++tensor_index) {
    const patchblob_tensor_contract_v2 *left_contract = &left->target.tensors[tensor_index];
    const patchblob_tensor_contract_v2 *right_contract = &right->target.tensors[tensor_index];
    const npunlock_tensor_routing_observation *left_routing = &left->tensors[tensor_index];
    const npunlock_tensor_routing_observation *right_routing = &right->tensors[tensor_index];
    int64_t current;
    if (left_contract->role != right_contract->role ||
        left_contract->tensor_index != right_contract->tensor_index ||
        left_contract->precision != right_contract->precision ||
        left_contract->element_count != right_contract->element_count ||
        left_contract->span_bytes != right_contract->span_bytes ||
        left_contract->observed_flags != right_contract->observed_flags ||
        left_routing->data_symbol != right_routing->data_symbol ||
        left_routing->rank != right_routing->rank ||
        memcmp(left_routing->dimensions, right_routing->dimensions,
               left_routing->rank * sizeof(left_routing->dimensions[0])) != 0 ||
        memcmp(left_routing->strides_bits, right_routing->strides_bits,
               left_routing->rank * sizeof(left_routing->strides_bits[0])) != 0 ||
        left_routing->data_addend < 0 || right_routing->data_addend < 0) {
      return false;
    }
    current = left_routing->data_addend >= right_routing->data_addend
                  ? left_routing->data_addend - right_routing->data_addend
                  : -(right_routing->data_addend - left_routing->data_addend);
    if (tensor_index == 0u) {
      result = current;
    } else if (current != result) {
      return false;
    }
  }
  if (result == 0) {
    return false;
  }
  *translation = result;
  return true;
}

static bool replica_translation_solves_group(
    const npunlock_invocation_routing_observation *invocations, size_t invocation_count,
    const npunlock_act_output_slice_candidate_observation *slice_candidates,
    size_t slice_candidate_count, uint32_t group_index, int64_t translation) {
  size_t index;
  for (index = 0; index < invocation_count; ++index) {
    const npunlock_act_output_slice_candidate_observation *slice;
    size_t slice_count;
    size_t match_count = 0u;
    size_t other_index;
    if (invocations[index].group_index != group_index) {
      continue;
    }
    slice_count =
        invocation_slice_candidate_count(slice_candidates, slice_candidate_count,
                                         invocations[index].target.invocation_index, &slice);
    if (slice_count != 0u) {
      continue;
    }
    for (other_index = 0; other_index < invocation_count; ++other_index) {
      int64_t candidate_translation;
      if (invocations[other_index].group_index != group_index ||
          invocation_slice_candidate_count(slice_candidates, slice_candidate_count,
                                           invocations[other_index].target.invocation_index,
                                           &slice) != 1u ||
          slice == NULL) {
        continue;
      }
      if (replica_contracts_match(&invocations[index], &invocations[other_index],
                                  &candidate_translation) &&
          candidate_translation == translation) {
        ++match_count;
      }
    }
    if (match_count != 1u) {
      return false;
    }
  }
  for (index = 0; index < invocation_count; ++index) {
    const npunlock_act_output_slice_candidate_observation *slice;
    size_t match_count = 0u;
    size_t other_index;
    if (invocations[index].group_index != group_index ||
        invocation_slice_candidate_count(slice_candidates, slice_candidate_count,
                                         invocations[index].target.invocation_index,
                                         &slice) != 1u ||
        slice == NULL) {
      continue;
    }
    for (other_index = 0; other_index < invocation_count; ++other_index) {
      const npunlock_act_output_slice_candidate_observation *other_slice;
      int64_t candidate_translation;
      if (invocations[other_index].group_index != group_index ||
          invocation_slice_candidate_count(slice_candidates, slice_candidate_count,
                                           invocations[other_index].target.invocation_index,
                                           &other_slice) != 0u ||
          !replica_contracts_match(&invocations[other_index], &invocations[index],
                                   &candidate_translation) ||
          candidate_translation != translation) {
        continue;
      }
      ++match_count;
    }
    if (match_count != 1u) {
      return false;
    }
  }
  return true;
}

npunlock_status npunlock_derive_act_replica_candidates(
    const npunlock_invocation_routing_observation *invocations, size_t invocation_count,
    size_t group_count, const npunlock_act_output_slice_candidate_observation *slice_candidates,
    size_t slice_candidate_count, npunlock_act_replica_group_observation **groups,
    size_t *replica_group_count, npunlock_act_replica_candidate_observation **candidates,
    size_t *replica_candidate_count, npunlock_diagnostic *diagnostic) {
  npunlock_act_replica_group_observation *group_results = NULL;
  npunlock_act_replica_candidate_observation *candidate_results = NULL;
  int64_t *translations = NULL;
  size_t allocation_size;
  size_t work_bound;
  size_t group_result_count = 0u;
  size_t candidate_result_count = 0u;
  size_t group_index;

  *groups = NULL;
  *replica_group_count = 0u;
  *candidates = NULL;
  *replica_candidate_count = 0u;
  if (!npunlock_checked_mul_size(invocation_count, invocation_count, &work_bound) ||
      !npunlock_checked_mul_size(work_bound, group_count, &work_bound) ||
      work_bound > ACT_EDGE_MAX_COMPARISONS ||
      !npunlock_checked_mul_size(group_count, sizeof(*group_results), &allocation_size)) {
    return unsupported(diagnostic, "ACT replica analysis exceeds its work bound");
  }
  group_results = (npunlock_act_replica_group_observation *)calloc(1u, allocation_size);
  if (!npunlock_checked_mul_size(invocation_count, sizeof(*candidate_results), &allocation_size)) {
    free(group_results);
    return unsupported(diagnostic, "ACT replica candidate allocation overflows");
  }
  candidate_results = (npunlock_act_replica_candidate_observation *)calloc(1u, allocation_size);
  if (!npunlock_checked_mul_size(invocation_count, invocation_count, &work_bound) ||
      !npunlock_checked_mul_size(work_bound, sizeof(*translations), &allocation_size)) {
    free(group_results);
    free(candidate_results);
    return unsupported(diagnostic, "ACT replica translation allocation overflows");
  }
  translations = (int64_t *)calloc(1u, allocation_size);
  if ((group_results == NULL && group_count != 0u) ||
      (candidate_results == NULL && invocation_count != 0u) ||
      (translations == NULL && invocation_count != 0u)) {
    free(group_results);
    free(candidate_results);
    free(translations);
    return npunlock_set_diagnostic(diagnostic, NPUNLOCK_STATUS_OUT_OF_MEMORY,
                                   "patchblob.inspect_routing",
                                   "could not allocate ACT replica observations");
  }

  for (group_index = 0; group_index < group_count; ++group_index) {
    size_t routed_count = 0u;
    size_t unrouted_count = 0u;
    size_t translation_count = 0u;
    size_t solution_count = 0u;
    int64_t solution = 0;
    size_t index;
    bool ambiguous_route = false;
    for (index = 0; index < invocation_count; ++index) {
      const npunlock_act_output_slice_candidate_observation *slice;
      size_t count;
      if (invocations[index].group_index != group_index) {
        continue;
      }
      count = invocation_slice_candidate_count(slice_candidates, slice_candidate_count,
                                               invocations[index].target.invocation_index, &slice);
      if (count == 0u) {
        ++unrouted_count;
      } else if (count == 1u && slice != NULL) {
        ++routed_count;
      } else {
        ambiguous_route = true;
      }
    }
    if (routed_count == 0u || unrouted_count == 0u) {
      continue;
    }
    for (index = 0; index < invocation_count && !ambiguous_route; ++index) {
      size_t other_index;
      const npunlock_act_output_slice_candidate_observation *slice;
      if (invocations[index].group_index != group_index ||
          invocation_slice_candidate_count(slice_candidates, slice_candidate_count,
                                           invocations[index].target.invocation_index,
                                           &slice) != 0u) {
        continue;
      }
      for (other_index = 0; other_index < invocation_count; ++other_index) {
        int64_t translation;
        size_t prior;
        bool unique = true;
        if (invocations[other_index].group_index != group_index ||
            invocation_slice_candidate_count(slice_candidates, slice_candidate_count,
                                             invocations[other_index].target.invocation_index,
                                             &slice) != 1u ||
            slice == NULL ||
            !replica_contracts_match(&invocations[index], &invocations[other_index],
                                     &translation)) {
          continue;
        }
        for (prior = 0; prior < translation_count; ++prior) {
          if (translations[prior] == translation) {
            unique = false;
            break;
          }
        }
        if (unique) {
          translations[translation_count++] = translation;
        }
      }
    }
    for (index = 0; index < translation_count; ++index) {
      if (replica_translation_solves_group(invocations, invocation_count, slice_candidates,
                                           slice_candidate_count, (uint32_t)group_index,
                                           translations[index])) {
        ++solution_count;
        solution = translations[index];
      }
    }
    group_results[group_result_count].act_group_index = (uint32_t)group_index;
    group_results[group_result_count].routed_invocation_count = (uint32_t)routed_count;
    group_results[group_result_count].unrouted_invocation_count = (uint32_t)unrouted_count;
    group_results[group_result_count].translation_solution_count = (uint32_t)solution_count;
    group_results[group_result_count].cmx_translation_candidate =
        solution_count == 1u ? solution : 0;
    ++group_result_count;
    if (solution_count != 1u) {
      continue;
    }
    for (index = 0; index < invocation_count; ++index) {
      const npunlock_act_output_slice_candidate_observation *slice;
      size_t other_index;
      if (invocations[index].group_index != group_index ||
          invocation_slice_candidate_count(slice_candidates, slice_candidate_count,
                                           invocations[index].target.invocation_index,
                                           &slice) != 0u) {
        continue;
      }
      for (other_index = 0; other_index < invocation_count; ++other_index) {
        int64_t translation;
        if (invocations[other_index].group_index != group_index ||
            invocation_slice_candidate_count(slice_candidates, slice_candidate_count,
                                             invocations[other_index].target.invocation_index,
                                             &slice) != 1u ||
            slice == NULL ||
            !replica_contracts_match(&invocations[index], &invocations[other_index],
                                     &translation) ||
            translation != solution) {
          continue;
        }
        candidate_results[candidate_result_count].act_group_index = (uint32_t)group_index;
        candidate_results[candidate_result_count].unrouted_invocation_index =
            invocations[index].target.invocation_index;
        candidate_results[candidate_result_count].routed_invocation_index =
            invocations[other_index].target.invocation_index;
        candidate_results[candidate_result_count].unrouted_tile_candidate_raw =
            invocations[index].tile_candidate_raw;
        candidate_results[candidate_result_count].routed_tile_candidate_raw =
            invocations[other_index].tile_candidate_raw;
        candidate_results[candidate_result_count].output_symbol_raw = slice->output_symbol_raw;
        candidate_results[candidate_result_count].output_slice_addend_candidate =
            slice->output_slice_addend_candidate;
        candidate_results[candidate_result_count].output_slice_span_bytes = slice->act_span_bytes;
        candidate_results[candidate_result_count].cmx_translation_candidate = solution;
        ++candidate_result_count;
        break;
      }
    }
  }
  free(translations);
  *groups = group_results;
  *replica_group_count = group_result_count;
  *candidates = candidate_results;
  *replica_candidate_count = candidate_result_count;
  return NPUNLOCK_STATUS_OK;
}

npunlock_status npunlock_derive_act_group_completeness(
    const npunlock_invocation_routing_observation *invocations, size_t invocation_count,
    size_t group_count, const npunlock_act_output_slice_candidate_observation *slice_candidates,
    size_t slice_candidate_count, const npunlock_act_output_cover_observation *covers,
    size_t cover_count, const npunlock_act_replica_group_observation *replica_groups,
    size_t replica_group_count,
    const npunlock_act_replica_candidate_observation *replica_candidates,
    size_t replica_candidate_count, npunlock_act_group_completeness_observation **completeness,
    size_t *completeness_count, npunlock_diagnostic *diagnostic) {
  npunlock_act_group_completeness_observation *result = NULL;
  size_t allocation_size;
  size_t result_count = 0u;
  size_t group_index;
  size_t work_bound;
  size_t work_term;

  if (completeness == NULL || completeness_count == NULL || diagnostic == NULL ||
      (invocations == NULL && invocation_count != 0u) ||
      (slice_candidates == NULL && slice_candidate_count != 0u) ||
      (covers == NULL && cover_count != 0u) ||
      (replica_groups == NULL && replica_group_count != 0u) ||
      (replica_candidates == NULL && replica_candidate_count != 0u)) {
    return NPUNLOCK_STATUS_INVALID_ARGUMENT;
  }
  *completeness = NULL;
  *completeness_count = 0u;
  if (invocation_count > UINT32_MAX || group_count > UINT32_MAX ||
      !npunlock_checked_add_size(replica_candidate_count, 1u, &work_term) ||
      !npunlock_checked_mul_size(invocation_count, work_term, &work_term) ||
      !npunlock_checked_add_size(work_term, slice_candidate_count, &work_bound) ||
      !npunlock_checked_add_size(work_bound, cover_count, &work_bound) ||
      !npunlock_checked_add_size(work_bound, replica_group_count, &work_bound) ||
      !npunlock_checked_mul_size(group_count, work_bound, &work_bound) ||
      work_bound > ACT_EDGE_MAX_COMPARISONS) {
    return unsupported(diagnostic, "ACT group-completeness analysis exceeds its work bound");
  }
  if (!npunlock_checked_mul_size(group_count, sizeof(*result), &allocation_size)) {
    return unsupported(diagnostic, "ACT group-completeness allocation overflows");
  }
  result = (npunlock_act_group_completeness_observation *)calloc(1u, allocation_size);
  if (result == NULL && group_count != 0u) {
    return npunlock_set_diagnostic(diagnostic, NPUNLOCK_STATUS_OUT_OF_MEMORY,
                                   "patchblob.inspect_routing",
                                   "could not allocate ACT group-completeness observations");
  }
  for (group_index = 0; group_index < group_count; ++group_index) {
    npunlock_act_group_completeness_observation *observation;
    const npunlock_act_replica_group_observation *replica_group = NULL;
    bool participates = false;
    bool output_symbol_set = false;
    size_t index;
    for (index = 0; index < slice_candidate_count; ++index) {
      if (slice_candidates[index].act_group_index == group_index) {
        participates = true;
        break;
      }
    }
    if (!participates) {
      continue;
    }
    observation = &result[result_count++];
    observation->act_group_index = (uint32_t)group_index;
    for (index = 0; index < replica_group_count; ++index) {
      if (replica_groups[index].act_group_index == group_index) {
        replica_group = &replica_groups[index];
        break;
      }
    }
    for (index = 0; index < invocation_count; ++index) {
      const npunlock_act_output_slice_candidate_observation *slice = NULL;
      size_t slice_count;
      size_t replica_count = 0u;
      size_t replica_index;
      if (invocations[index].group_index != group_index) {
        continue;
      }
      ++observation->invocation_count;
      slice_count =
          invocation_slice_candidate_count(slice_candidates, slice_candidate_count,
                                           invocations[index].target.invocation_index, &slice);
      if (slice_count == 1u && slice != NULL) {
        ++observation->routed_invocation_count;
        if (!output_symbol_set) {
          observation->output_symbol_raw = slice->output_symbol_raw;
          output_symbol_set = true;
        } else if (observation->output_symbol_raw != slice->output_symbol_raw) {
          observation->unresolved_flags |= NPUNLOCK_GROUP_COMPLETE_OUTPUT_SYMBOL_AMBIGUOUS;
        }
        continue;
      }
      if (slice_count != 0u) {
        ++observation->unresolved_invocation_count;
        observation->unresolved_flags |= NPUNLOCK_GROUP_COMPLETE_ROUTE_AMBIGUOUS;
        continue;
      }
      for (replica_index = 0; replica_index < replica_candidate_count; ++replica_index) {
        if (replica_candidates[replica_index].act_group_index == group_index &&
            replica_candidates[replica_index].unrouted_invocation_index ==
                invocations[index].target.invocation_index) {
          ++replica_count;
        }
      }
      if (replica_count == 1u) {
        ++observation->replica_invocation_count;
      } else {
        ++observation->unresolved_invocation_count;
        observation->unresolved_flags |=
            replica_group != NULL && replica_group->translation_solution_count != 1u
                ? NPUNLOCK_GROUP_COMPLETE_REPLICA_UNRESOLVED
                : NPUNLOCK_GROUP_COMPLETE_INVOCATION_UNCLASSIFIED;
      }
    }
    if (!output_symbol_set) {
      observation->unresolved_flags |= NPUNLOCK_GROUP_COMPLETE_OUTPUT_SYMBOL_AMBIGUOUS;
    } else {
      const npunlock_act_output_cover_observation *cover = NULL;
      size_t matching_covers = 0u;
      for (index = 0; index < cover_count; ++index) {
        if (covers[index].output_symbol_raw == observation->output_symbol_raw) {
          cover = &covers[index];
          ++matching_covers;
        }
      }
      if (matching_covers != 1u || cover == NULL || !cover->exact_cover) {
        observation->unresolved_flags |= NPUNLOCK_GROUP_COMPLETE_OUTPUT_COVER_UNRESOLVED;
      }
    }
    observation->complete =
        observation->unresolved_flags == 0u && observation->unresolved_invocation_count == 0u &&
        observation->routed_invocation_count + observation->replica_invocation_count ==
            observation->invocation_count;
  }
  *completeness = result;
  *completeness_count = result_count;
  return NPUNLOCK_STATUS_OK;
}

void npunlock_graph_routing_observations_release(
    npunlock_graph_routing_observations *observations) {
  if (observations == NULL) {
    return;
  }
  free(observations->invocations);
  free(observations->network_output_routes);
  free(observations->act_output_slice_candidates);
  free(observations->act_output_covers);
  free(observations->act_replica_groups);
  free(observations->act_replica_candidates);
  free(observations->act_group_completeness);
  free(observations->act_edge_candidates);
  free(observations->dpu_invariants);
  free(observations->act_dpu_edge_candidates);
  free(observations->act_dpu_output_covers);
  memset(observations, 0, sizeof(*observations));
}

static bool observation_arrays_equal(const void *left, size_t left_count, const void *right,
                                     size_t right_count, size_t element_size) {
  size_t byte_count;
  if (left_count != right_count || (left == NULL && left_count != 0u) ||
      (right == NULL && right_count != 0u) ||
      !npunlock_checked_mul_size(left_count, element_size, &byte_count)) {
    return false;
  }
  return byte_count == 0u || memcmp(left, right, byte_count) == 0;
}

npunlock_status
npunlock_compare_graph_routing_observations(const npunlock_graph_routing_observations *baseline,
                                            const npunlock_graph_routing_observations *candidate,
                                            npunlock_graph_routing_comparison *comparison,
                                            npunlock_diagnostic *diagnostic) {
  uint8_t *changed = NULL;
  uint8_t *unchanged = NULL;
  uint32_t *changed_groups = NULL;
  size_t changed_group_count = 0u;
  size_t changed_group_bytes;
  size_t index;

  if (baseline == NULL || candidate == NULL || comparison == NULL || diagnostic == NULL ||
      (baseline->invocations == NULL && baseline->invocation_count != 0u) ||
      (candidate->invocations == NULL && candidate->invocation_count != 0u)) {
    return NPUNLOCK_STATUS_INVALID_ARGUMENT;
  }
  memset(comparison, 0, sizeof(*comparison));
  if (baseline->group_count != candidate->group_count ||
      baseline->invocation_count != candidate->invocation_count ||
      baseline->tile_count_observed != candidate->tile_count_observed) {
    comparison->mismatch_flags |= NPUNLOCK_COMPARE_STRUCTURE_CHANGED;
  } else {
    if (!npunlock_checked_mul_size(baseline->group_count, sizeof(*changed_groups),
                                   &changed_group_bytes)) {
      return unsupported(diagnostic, "graph-comparison group allocation overflows");
    }
    changed = (uint8_t *)calloc(baseline->group_count, sizeof(*changed));
    unchanged = (uint8_t *)calloc(baseline->group_count, sizeof(*unchanged));
    changed_groups = (uint32_t *)malloc(changed_group_bytes);
    if ((changed == NULL || unchanged == NULL || changed_groups == NULL) &&
        baseline->group_count != 0u) {
      free(changed);
      free(unchanged);
      free(changed_groups);
      return npunlock_set_diagnostic(diagnostic, NPUNLOCK_STATUS_OUT_OF_MEMORY,
                                     "patchblob.compare_routing",
                                     "could not allocate graph-comparison group state");
    }
    for (index = 0; index < baseline->invocation_count; ++index) {
      npunlock_invocation_routing_observation left;
      npunlock_invocation_routing_observation right;
      uint32_t group_index = baseline->invocations[index].group_index;
      bool code_equal;
      if (group_index >= baseline->group_count ||
          candidate->invocations[index].group_index != group_index) {
        comparison->mismatch_flags |= NPUNLOCK_COMPARE_STRUCTURE_CHANGED;
        continue;
      }
      code_equal =
          baseline->invocations[index].code_extent == candidate->invocations[index].code_extent &&
          memcmp(baseline->invocations[index].code_sha256,
                 candidate->invocations[index].code_sha256,
                 sizeof(baseline->invocations[index].code_sha256)) == 0;
      if (code_equal) {
        unchanged[group_index] = 1u;
      } else {
        changed[group_index] = 1u;
      }
      left = baseline->invocations[index];
      right = candidate->invocations[index];
      left.code_offset_raw = 0u;
      right.code_offset_raw = 0u;
      left.code_extent = 0u;
      right.code_extent = 0u;
      memset(left.code_sha256, 0, sizeof(left.code_sha256));
      memset(right.code_sha256, 0, sizeof(right.code_sha256));
      left.target.range_index = 0u;
      right.target.range_index = 0u;
      if (memcmp(&left, &right, sizeof(left)) != 0) {
        comparison->mismatch_flags |= NPUNLOCK_COMPARE_INVOCATION_CONTRACT_CHANGED;
      }
    }
    for (index = 0; index < baseline->group_count; ++index) {
      if (changed[index] != 0u) {
        changed_groups[changed_group_count++] = (uint32_t)index;
        if (unchanged[index] != 0u) {
          comparison->mismatch_flags |= NPUNLOCK_COMPARE_PARTIAL_GROUP_CODE_CHANGE;
        }
      }
    }
  }

  if (!observation_arrays_equal(
          baseline->network_output_routes, baseline->network_output_route_count,
          candidate->network_output_routes, candidate->network_output_route_count,
          sizeof(*baseline->network_output_routes))) {
    comparison->mismatch_flags |= NPUNLOCK_COMPARE_OUTPUT_ROUTING_CHANGED;
  }
  if (!observation_arrays_equal(
          baseline->act_output_slice_candidates, baseline->act_output_slice_candidate_count,
          candidate->act_output_slice_candidates, candidate->act_output_slice_candidate_count,
          sizeof(*baseline->act_output_slice_candidates)) ||
      !observation_arrays_equal(baseline->act_output_covers, baseline->act_output_cover_count,
                                candidate->act_output_covers, candidate->act_output_cover_count,
                                sizeof(*baseline->act_output_covers)) ||
      !observation_arrays_equal(baseline->act_replica_groups, baseline->act_replica_group_count,
                                candidate->act_replica_groups, candidate->act_replica_group_count,
                                sizeof(*baseline->act_replica_groups)) ||
      !observation_arrays_equal(
          baseline->act_replica_candidates, baseline->act_replica_candidate_count,
          candidate->act_replica_candidates, candidate->act_replica_candidate_count,
          sizeof(*baseline->act_replica_candidates)) ||
      !observation_arrays_equal(
          baseline->act_group_completeness, baseline->act_group_completeness_count,
          candidate->act_group_completeness, candidate->act_group_completeness_count,
          sizeof(*baseline->act_group_completeness)) ||
      !observation_arrays_equal(baseline->act_edge_candidates, baseline->act_edge_candidate_count,
                                candidate->act_edge_candidates, candidate->act_edge_candidate_count,
                                sizeof(*baseline->act_edge_candidates))) {
    comparison->mismatch_flags |= NPUNLOCK_COMPARE_ACT_RELATIONS_CHANGED;
  }
  if (!observation_arrays_equal(baseline->dpu_invariants, baseline->dpu_invariant_count,
                                candidate->dpu_invariants, candidate->dpu_invariant_count,
                                sizeof(*baseline->dpu_invariants))) {
    comparison->mismatch_flags |= NPUNLOCK_COMPARE_DPU_CHANGED;
  }
  if (!observation_arrays_equal(
          baseline->act_dpu_edge_candidates, baseline->act_dpu_edge_candidate_count,
          candidate->act_dpu_edge_candidates, candidate->act_dpu_edge_candidate_count,
          sizeof(*baseline->act_dpu_edge_candidates)) ||
      !observation_arrays_equal(
          baseline->act_dpu_output_covers, baseline->act_dpu_output_cover_count,
          candidate->act_dpu_output_covers, candidate->act_dpu_output_cover_count,
          sizeof(*baseline->act_dpu_output_covers))) {
    comparison->mismatch_flags |= NPUNLOCK_COMPARE_ACT_DPU_RELATIONS_CHANGED;
  }
  free(changed);
  free(unchanged);
  comparison->changed_group_indices = changed_groups;
  comparison->changed_group_count = changed_group_count;
  comparison->contract_stable = comparison->mismatch_flags == 0u;
  comparison->unique_changed_group = changed_group_count == 1u;
  return NPUNLOCK_STATUS_OK;
}

void npunlock_graph_routing_comparison_release(npunlock_graph_routing_comparison *comparison) {
  if (comparison == NULL) {
    return;
  }
  free(comparison->changed_group_indices);
  memset(comparison, 0, sizeof(*comparison));
}

npunlock_status npunlock_inspect_graph_routing(npunlock_view graph_blob,
                                               npunlock_graph_routing_observations *observations,
                                               npunlock_diagnostic *diagnostic) {
  graph_layout layout;
  patchblob_discovered_target_v2 *discovered = NULL;
  npunlock_invocation_routing_observation *invocations = NULL;
  npunlock_network_output_route_observation *network_output_routes = NULL;
  npunlock_act_output_slice_candidate_observation *act_output_slice_candidates = NULL;
  npunlock_act_output_cover_observation *act_output_covers = NULL;
  npunlock_act_replica_group_observation *act_replica_groups = NULL;
  npunlock_act_replica_candidate_observation *act_replica_candidates = NULL;
  npunlock_act_group_completeness_observation *act_group_completeness = NULL;
  npunlock_act_edge_candidate_observation *act_edge_candidates = NULL;
  npunlock_dpu_invariant_observation *dpu_invariants = NULL;
  npunlock_act_dpu_edge_candidate_observation *act_dpu_edge_candidates = NULL;
  npunlock_act_dpu_output_cover_observation *act_dpu_output_covers = NULL;
  uint64_t *parameter_bases = NULL;
  size_t target_count = 0;
  size_t group_count = 0;
  size_t invocation_count = 0;
  size_t network_output_route_count = 0;
  size_t act_output_slice_candidate_count = 0;
  size_t act_output_cover_count = 0;
  size_t act_replica_group_count = 0;
  size_t act_replica_candidate_count = 0;
  size_t act_group_completeness_count = 0;
  size_t act_edge_candidate_count = 0;
  size_t dpu_invariant_count = 0;
  size_t act_dpu_edge_candidate_count = 0;
  size_t act_dpu_output_cover_count = 0;
  size_t index;
  uint32_t tile_count_observed = 0;
  npunlock_status status;

  if (observations == NULL || diagnostic == NULL) {
    return NPUNLOCK_STATUS_INVALID_ARGUMENT;
  }
  memset(observations, 0, sizeof(*observations));
  memset(&layout, 0, sizeof(layout));
  status = npunlock_discover_graph_targets_v2(graph_blob, &discovered, &target_count, &group_count,
                                              diagnostic);
  if (status != NPUNLOCK_STATUS_OK) {
    return status;
  }
  status = parse_graph(graph_blob, &layout, diagnostic);
  if (status != NPUNLOCK_STATUS_OK) {
    goto fail;
  }
  status = invocation_parameter_bases(&layout, &parameter_bases, &invocation_count, diagnostic);
  if (status != NPUNLOCK_STATUS_OK) {
    goto fail;
  }
  if (invocation_count != target_count) {
    status = unsupported(diagnostic, "routing observation count does not match ACT discovery");
    goto fail;
  }
  status = inspect_network_output_routes(&layout, &network_output_routes,
                                         &network_output_route_count, diagnostic);
  if (status != NPUNLOCK_STATUS_OK) {
    goto fail;
  }
  status = inspect_dpu_invariants(&layout, &dpu_invariants, &dpu_invariant_count, diagnostic);
  if (status != NPUNLOCK_STATUS_OK) {
    goto fail;
  }
  tile_count_observed = layout.tile_count;
  invocations =
      (npunlock_invocation_routing_observation *)calloc(invocation_count, sizeof(*invocations));
  if (invocations == NULL && invocation_count != 0u) {
    status = npunlock_set_diagnostic(diagnostic, NPUNLOCK_STATUS_OUT_OF_MEMORY,
                                     "patchblob.inspect_routing",
                                     "could not allocate routing observations");
    goto fail;
  }
  for (index = 0; index < invocation_count; ++index) {
    const graph_section *invocation_section = &layout.sections[layout.invocations_index];
    const graph_section *range_section = &layout.sections[layout.ranges_index];
    const graph_section *text_section = &layout.sections[layout.kernel_text_index];
    const uint8_t *record =
        layout.blob.data + invocation_section->offset + index * ACT_INVOCATION_SIZE;
    npunlock_invocation_routing_observation *observation = &invocations[index];
    graph_relocation range_relocation;
    size_t code_offset;
    size_t range_offset;
    size_t tensor_index;

    observation->group_index = discovered[index].group_index;
    status = find_range_relocation(&layout, discovered[index].target.range_index, &range_relocation,
                                   diagnostic);
    if (status != NPUNLOCK_STATUS_OK) {
      goto fail;
    }
    range_offset =
        range_section->offset + (size_t)discovered[index].target.range_index * ACT_RANGE_SIZE;
    observation->code_extent = read_u32(layout.blob.data + range_offset + 0x0cu);
    if (range_relocation.addend < 0 ||
        !u64_to_size((uint64_t)range_relocation.addend, &code_offset) ||
        !range_in_bounds(code_offset, observation->code_extent, text_section->size)) {
      status = malformed(diagnostic, "ACT range code slice is outside KernelText");
      goto fail;
    }
    observation->code_offset_raw = (uint64_t)code_offset;
    npunlock_sha256((npunlock_view){layout.blob.data + text_section->offset + code_offset,
                                    observation->code_extent},
                    observation->code_sha256);
    observation->tile_candidate_raw = read_u32(record + 0x34u);
    observation->parameter_base = parameter_bases[index];
    observation->wait_mask_raw = read_u64(record + 0x10u);
    observation->post_mask_raw = read_u64(record + 0x18u);
    observation->barrier_group_raw = record[0x20u];
    observation->barrier_mask_raw = record[0x21u];
    observation->start_after_raw = read_u32(record + 0x28u);
    observation->clean_after_raw = read_u32(record + 0x2cu);
    observation->target = discovered[index].target;
    for (tensor_index = 0; tensor_index < observation->target.tensor_count; ++tensor_index) {
      memref_contract contract;
      npunlock_tensor_routing_observation *tensor = &observation->tensors[tensor_index];
      status = validate_memref(&layout,
                               observation->parameter_base + (uint64_t)tensor_index * MEMREF_SIZE,
                               &contract, diagnostic);
      if (status != NPUNLOCK_STATUS_OK) {
        goto fail;
      }
      tensor->data_symbol = contract.data_symbol;
      tensor->data_addend = contract.data_addend;
      tensor->rank = contract.rank;
      memcpy(tensor->dimensions, contract.dimensions,
             (size_t)contract.rank * sizeof(contract.dimensions[0]));
      memcpy(tensor->strides_bits, contract.strides_bits,
             (size_t)contract.rank * sizeof(contract.strides_bits[0]));
    }
  }
  status = npunlock_derive_act_edge_candidates(invocations, invocation_count, &act_edge_candidates,
                                               &act_edge_candidate_count, diagnostic);
  if (status != NPUNLOCK_STATUS_OK) {
    goto fail;
  }
  status = derive_act_output_slice_candidates(
      invocations, invocation_count, network_output_routes, network_output_route_count,
      &act_output_slice_candidates, &act_output_slice_candidate_count, diagnostic);
  if (status != NPUNLOCK_STATUS_OK) {
    goto fail;
  }
  status = npunlock_derive_act_output_cover_observations(
      invocations, invocation_count, network_output_routes, network_output_route_count,
      act_output_slice_candidates, act_output_slice_candidate_count, &act_output_covers,
      &act_output_cover_count, diagnostic);
  if (status != NPUNLOCK_STATUS_OK) {
    goto fail;
  }
  status = npunlock_derive_act_replica_candidates(
      invocations, invocation_count, group_count, act_output_slice_candidates,
      act_output_slice_candidate_count, &act_replica_groups, &act_replica_group_count,
      &act_replica_candidates, &act_replica_candidate_count, diagnostic);
  if (status != NPUNLOCK_STATUS_OK) {
    goto fail;
  }
  status = npunlock_derive_act_group_completeness(
      invocations, invocation_count, group_count, act_output_slice_candidates,
      act_output_slice_candidate_count, act_output_covers, act_output_cover_count,
      act_replica_groups, act_replica_group_count, act_replica_candidates,
      act_replica_candidate_count, &act_group_completeness, &act_group_completeness_count,
      diagnostic);
  if (status != NPUNLOCK_STATUS_OK) {
    goto fail;
  }
  status = derive_act_dpu_edge_candidates(invocations, invocation_count, dpu_invariants,
                                          dpu_invariant_count, &act_dpu_edge_candidates,
                                          &act_dpu_edge_candidate_count, diagnostic);
  if (status != NPUNLOCK_STATUS_OK) {
    goto fail;
  }
  status = npunlock_derive_act_dpu_output_cover_observations(
      invocations, invocation_count, group_count, act_dpu_edge_candidates,
      act_dpu_edge_candidate_count, &act_dpu_output_covers, &act_dpu_output_cover_count,
      diagnostic);
  if (status != NPUNLOCK_STATUS_OK) {
    goto fail;
  }

  free(discovered);
  free(parameter_bases);
  graph_layout_release(&layout);
  observations->invocations = invocations;
  observations->invocation_count = invocation_count;
  observations->group_count = group_count;
  observations->tile_count_observed = tile_count_observed;
  observations->network_output_routes = network_output_routes;
  observations->network_output_route_count = network_output_route_count;
  observations->act_output_slice_candidates = act_output_slice_candidates;
  observations->act_output_slice_candidate_count = act_output_slice_candidate_count;
  observations->act_output_covers = act_output_covers;
  observations->act_output_cover_count = act_output_cover_count;
  observations->act_replica_groups = act_replica_groups;
  observations->act_replica_group_count = act_replica_group_count;
  observations->act_replica_candidates = act_replica_candidates;
  observations->act_replica_candidate_count = act_replica_candidate_count;
  observations->act_group_completeness = act_group_completeness;
  observations->act_group_completeness_count = act_group_completeness_count;
  observations->act_edge_candidates = act_edge_candidates;
  observations->act_edge_candidate_count = act_edge_candidate_count;
  observations->dpu_invariants = dpu_invariants;
  observations->dpu_invariant_count = dpu_invariant_count;
  observations->act_dpu_edge_candidates = act_dpu_edge_candidates;
  observations->act_dpu_edge_candidate_count = act_dpu_edge_candidate_count;
  observations->act_dpu_output_covers = act_dpu_output_covers;
  observations->act_dpu_output_cover_count = act_dpu_output_cover_count;
  return NPUNLOCK_STATUS_OK;

fail:
  free(invocations);
  free(network_output_routes);
  free(act_output_slice_candidates);
  free(act_output_covers);
  free(act_replica_groups);
  free(act_replica_candidates);
  free(act_group_completeness);
  free(act_edge_candidates);
  free(dpu_invariants);
  free(act_dpu_edge_candidates);
  free(act_dpu_output_covers);
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

static npunlock_status patch_graph_blob_impl(npunlock_view graph_blob, npunlock_view image,
                                             const patchblob_target *targets,
                                             const patchblob_target_v2 *targets_v2,
                                             size_t target_count, uint32_t image_alignment,
                                             uint32_t tail_padding, npunlock_buffer *output,
                                             npunlock_patch_summary *summary,
                                             npunlock_diagnostic *diagnostic) {
  graph_layout layout;
  graph_layout patched_layout;
  uint64_t *parameter_bases = NULL;
  size_t invocation_count = 0;
  const graph_section *text;
  size_t image_base;
  size_t image_end;
  size_t new_text_size;
  size_t minimum_inserted_size;
  size_t inserted_size;
  size_t shift_alignment = 1u;
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
  summary->tile_count = layout.tile_count;
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
    uint32_t invocation_index =
        targets_v2 != NULL ? targets_v2[index].invocation_index : targets[index].invocation_index;
    uint32_t range_index =
        targets_v2 != NULL ? targets_v2[index].range_index : targets[index].range_index;
    size_t other;
    for (other = 0; other < index; ++other) {
      uint32_t other_invocation =
          targets_v2 != NULL ? targets_v2[other].invocation_index : targets[other].invocation_index;
      uint32_t other_range =
          targets_v2 != NULL ? targets_v2[other].range_index : targets[other].range_index;
      if (other_invocation == invocation_index || other_range == range_index) {
        status = unsupported(diagnostic, "target invocation and range selections must be unique");
        goto fail;
      }
    }
    status = targets_v2 != NULL
                 ? validate_target_v2(&layout, parameter_bases, invocation_count,
                                      &targets_v2[index], &summary->details[index], diagnostic)
                 : validate_target(&layout, parameter_bases, invocation_count, &targets[index],
                                   &summary->details[index], diagnostic);
    if (status != NPUNLOCK_STATUS_OK) {
      goto fail;
    }
    status = find_range_relocation(&layout, range_index, &relocation, diagnostic);
    if (status != NPUNLOCK_STATUS_OK) {
      goto fail;
    }
    summary->details[index].addend_file_offset = relocation.file_offset + 16u;
    summary->details[index].old_addend = relocation.addend;
  }
  if (targets_v2 != NULL) {
    for (index = 0; index < target_count; ++index) {
      size_t invocation_index;
      if ((targets_v2[index].target_flags & PATCHBLOB_TARGET_UNARY_CONVERSION_V2) == 0u) {
        continue;
      }
      for (invocation_index = 0; invocation_index < invocation_count; ++invocation_index) {
        size_t selected_index;
        bool selected = false;
        if (!invocation_group_identity_equal(&layout, targets_v2[index].invocation_index,
                                             invocation_index)) {
          continue;
        }
        for (selected_index = 0; selected_index < target_count; ++selected_index) {
          if (targets_v2[selected_index].invocation_index == invocation_index) {
            selected = true;
            break;
          }
        }
        if (!selected) {
          status = unsupported(
              diagnostic,
              "version 2 conversion targets must include the complete ACT replica group");
          goto fail;
        }
      }
    }
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
    if (section->offset >= summary->insertion_file_offset && section->offset != 0u &&
        section->alignment > shift_alignment) {
      shift_alignment = section->alignment;
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
    if (layout.section_table_offset >= summary->insertion_file_offset &&
        ELF64_SECTION_ALIGNMENT > shift_alignment) {
      shift_alignment = ELF64_SECTION_ALIGNMENT;
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
  minimum_inserted_size = new_text_size - text->size;
  if (!align_up(minimum_inserted_size, shift_alignment, &inserted_size) ||
      !npunlock_checked_add_size(text->size, inserted_size, &new_text_size)) {
    status = npunlock_set_diagnostic(diagnostic, NPUNLOCK_STATUS_OVERFLOW, "patchblob.patch",
                                     "appended code image size overflows");
    goto fail;
  }
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

npunlock_status npunlock_patch_graph_blob(npunlock_view graph_blob, npunlock_view image,
                                          const patchblob_target *targets, size_t target_count,
                                          uint32_t image_alignment, uint32_t tail_padding,
                                          npunlock_buffer *output, npunlock_patch_summary *summary,
                                          npunlock_diagnostic *diagnostic) {
  return patch_graph_blob_impl(graph_blob, image, targets, NULL, target_count, image_alignment,
                               tail_padding, output, summary, diagnostic);
}

npunlock_status npunlock_patch_graph_blob_v2(npunlock_view graph_blob, npunlock_view image,
                                             const patchblob_target_v2 *targets,
                                             size_t target_count, uint32_t image_alignment,
                                             uint32_t tail_padding, npunlock_buffer *output,
                                             npunlock_patch_summary *summary,
                                             npunlock_diagnostic *diagnostic) {
  return patch_graph_blob_impl(graph_blob, image, NULL, targets, target_count, image_alignment,
                               tail_padding, output, summary, diagnostic);
}
