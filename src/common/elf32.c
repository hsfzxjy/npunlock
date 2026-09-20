#include "elf32.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "internal.h"

enum {
  ELF32_HEADER_SIZE = 52,
  ELF32_SECTION_SIZE = 40,
  ELF32_PROGRAM_SIZE = 32,
  ELF32_SYMBOL_SIZE = 16,
  ELF_TYPE_EXEC = 2,
  ELF_MACHINE_SPARC = 2,
  ELF_SECTION_PROGBITS = 1,
  ELF_SECTION_SYMTAB = 2,
  ELF_SECTION_STRTAB = 3,
  ELF_SECTION_RELA = 4,
  ELF_SECTION_NOBITS = 8,
  ELF_SECTION_REL = 9,
  ELF_PROGRAM_LOAD = 1,
  ELF_SHN_UNDEF = 0,
  SHAVE_TEXT_ADDRESS = 0x1d000000,
  SHAVE_ARG_ADDRESS = 0x1e000000
};

typedef struct elf32_section {
  uint32_t name;
  uint32_t type;
  uint32_t flags;
  uint32_t address;
  uint32_t offset;
  uint32_t size;
  uint32_t link;
  uint32_t entry_size;
} elf32_section;

static uint16_t load_u16(const uint8_t *data) {
  return (uint16_t)((uint16_t)data[0] | ((uint16_t)data[1] << 8));
}

static uint32_t load_u32(const uint8_t *data) {
  return (uint32_t)data[0] | ((uint32_t)data[1] << 8) | ((uint32_t)data[2] << 16) |
         ((uint32_t)data[3] << 24);
}

static bool range_is_valid(size_t total, size_t offset, size_t size) {
  return offset <= total && size <= total - offset;
}

static npunlock_status fail(npunlock_diagnostic *diagnostic, npunlock_status status,
                            const char *message) {
  if (diagnostic == NULL) {
    return status;
  }
  return npunlock_set_diagnostic(diagnostic, status, "elf32.validate", message);
}

static bool read_section(npunlock_view elf, size_t table_offset, uint16_t entry_size,
                         uint16_t index, elf32_section *section) {
  size_t relative;
  size_t offset;
  const uint8_t *data;
  if (!npunlock_checked_mul_size(index, entry_size, &relative) ||
      !npunlock_checked_add_size(table_offset, relative, &offset) ||
      !range_is_valid(elf.size, offset, ELF32_SECTION_SIZE)) {
    return false;
  }
  data = elf.data + offset;
  section->name = load_u32(data);
  section->type = load_u32(data + 4);
  section->flags = load_u32(data + 8);
  section->address = load_u32(data + 12);
  section->offset = load_u32(data + 16);
  section->size = load_u32(data + 20);
  section->link = load_u32(data + 24);
  section->entry_size = load_u32(data + 36);
  return true;
}

static bool string_at(npunlock_view elf, const elf32_section *strings, uint32_t offset,
                      const char **value) {
  const uint8_t *start;
  size_t remaining;
  if (offset >= strings->size || !range_is_valid(elf.size, strings->offset, strings->size)) {
    return false;
  }
  start = elf.data + strings->offset + offset;
  remaining = strings->size - offset;
  if (memchr(start, 0, remaining) == NULL) {
    return false;
  }
  *value = (const char *)start;
  return true;
}

static npunlock_status validate_symbols(npunlock_view elf, size_t section_table,
                                        uint16_t section_entry_size, uint16_t section_count,
                                        const elf32_section *symbols,
                                        npunlock_diagnostic *diagnostic) {
  elf32_section strings;
  uint32_t index;
  if (symbols->entry_size != ELF32_SYMBOL_SIZE || symbols->size % ELF32_SYMBOL_SIZE != 0 ||
      symbols->link >= section_count ||
      !read_section(elf, section_table, section_entry_size, (uint16_t)symbols->link, &strings) ||
      strings.type != ELF_SECTION_STRTAB) {
    return fail(diagnostic, NPUNLOCK_STATUS_MALFORMED_INPUT, "invalid ELF32 symbol table metadata");
  }
  for (index = 1; index < symbols->size / ELF32_SYMBOL_SIZE; ++index) {
    size_t relative;
    size_t offset;
    const uint8_t *symbol;
    uint32_t name;
    uint16_t section_index;
    const char *symbol_name;
    if (!npunlock_checked_mul_size(index, ELF32_SYMBOL_SIZE, &relative) ||
        !npunlock_checked_add_size(symbols->offset, relative, &offset) ||
        !range_is_valid(elf.size, offset, ELF32_SYMBOL_SIZE)) {
      return fail(diagnostic, NPUNLOCK_STATUS_MALFORMED_INPUT,
                  "ELF32 symbol lies outside the file");
    }
    symbol = elf.data + offset;
    name = load_u32(symbol);
    section_index = load_u16(symbol + 14);
    if (name != 0 && !string_at(elf, &strings, name, &symbol_name)) {
      return fail(diagnostic, NPUNLOCK_STATUS_MALFORMED_INPUT, "ELF32 symbol name is invalid");
    }
    if (name != 0 && section_index == ELF_SHN_UNDEF) {
      (void)symbol_name;
      return fail(diagnostic, NPUNLOCK_STATUS_UNSUPPORTED,
                  "SHAVE ELF contains an undefined symbol");
    }
  }
  return NPUNLOCK_STATUS_OK;
}

npunlock_status npunlock_parse_shave_elf(npunlock_view elf, npunlock_shave_image *image,
                                         npunlock_diagnostic *diagnostic) {
  uint32_t program_table;
  uint32_t section_table;
  uint32_t entry;
  uint16_t program_entry_size;
  uint16_t program_count;
  uint16_t section_entry_size;
  uint16_t section_count;
  uint16_t string_index;
  elf32_section strings;
  elf32_section text = {0};
  elf32_section arg_data = {0};
  bool have_text = false;
  bool have_arg_data = false;
  bool text_is_loadable = false;
  uint16_t index;
  size_t table_size;

  if (image == NULL || !npunlock_view_is_valid(elf)) {
    return fail(diagnostic, NPUNLOCK_STATUS_INVALID_ARGUMENT, "invalid ELF32 input view");
  }
  memset(image, 0, sizeof(*image));
  if (elf.size < ELF32_HEADER_SIZE ||
      memcmp(elf.data,
             "\x7f"
             "ELF",
             4) != 0 ||
      elf.data[4] != 1 || elf.data[5] != 1 || elf.data[6] != 1) {
    return fail(diagnostic, NPUNLOCK_STATUS_MALFORMED_INPUT,
                "expected ELF32 little-endian version 1");
  }
  if (load_u16(elf.data + 16) != ELF_TYPE_EXEC || load_u16(elf.data + 18) != ELF_MACHINE_SPARC ||
      load_u32(elf.data + 20) != 1) {
    return fail(diagnostic, NPUNLOCK_STATUS_UNSUPPORTED, "expected an executable SPARC/SHAVE ELF");
  }
  entry = load_u32(elf.data + 24);
  program_table = load_u32(elf.data + 28);
  section_table = load_u32(elf.data + 32);
  program_entry_size = load_u16(elf.data + 42);
  program_count = load_u16(elf.data + 44);
  section_entry_size = load_u16(elf.data + 46);
  section_count = load_u16(elf.data + 48);
  string_index = load_u16(elf.data + 50);
  if (load_u16(elf.data + 40) < ELF32_HEADER_SIZE ||
      (program_count != 0 && program_entry_size < ELF32_PROGRAM_SIZE) ||
      section_entry_size < ELF32_SECTION_SIZE || section_count == 0 ||
      string_index >= section_count ||
      !npunlock_checked_mul_size(section_entry_size, section_count, &table_size) ||
      !range_is_valid(elf.size, section_table, table_size)) {
    return fail(diagnostic, NPUNLOCK_STATUS_MALFORMED_INPUT,
                "invalid ELF32 header or section table");
  }
  if (program_count != 0 &&
      (!npunlock_checked_mul_size(program_entry_size, program_count, &table_size) ||
       !range_is_valid(elf.size, program_table, table_size))) {
    return fail(diagnostic, NPUNLOCK_STATUS_MALFORMED_INPUT, "invalid ELF32 program table");
  }
  if (!read_section(elf, section_table, section_entry_size, string_index, &strings) ||
      strings.type != ELF_SECTION_STRTAB ||
      !range_is_valid(elf.size, strings.offset, strings.size)) {
    return fail(diagnostic, NPUNLOCK_STATUS_MALFORMED_INPUT, "invalid ELF32 section-name table");
  }

  for (index = 0; index < section_count; ++index) {
    elf32_section section;
    const char *name;
    npunlock_status status;
    if (!read_section(elf, section_table, section_entry_size, index, &section) ||
        (section.type != ELF_SECTION_NOBITS &&
         !range_is_valid(elf.size, section.offset, section.size)) ||
        !string_at(elf, &strings, section.name, &name)) {
      return fail(diagnostic, NPUNLOCK_STATUS_MALFORMED_INPUT,
                  "ELF32 section is malformed or out of bounds");
    }
    if ((section.type == ELF_SECTION_REL || section.type == ELF_SECTION_RELA) &&
        section.size != 0) {
      return fail(diagnostic, NPUNLOCK_STATUS_UNSUPPORTED,
                  "SHAVE ELF contains unresolved relocations");
    }
    if (section.type == ELF_SECTION_SYMTAB) {
      status = validate_symbols(elf, section_table, section_entry_size, section_count, &section,
                                diagnostic);
      if (status != NPUNLOCK_STATUS_OK) {
        return status;
      }
    }
    if (strcmp(name, ".text") == 0) {
      if (have_text) {
        return fail(diagnostic, NPUNLOCK_STATUS_MALFORMED_INPUT,
                    "SHAVE ELF has multiple .text sections");
      }
      text = section;
      have_text = true;
    } else if (strcmp(name, ".arg.data") == 0) {
      if (have_arg_data) {
        return fail(diagnostic, NPUNLOCK_STATUS_MALFORMED_INPUT,
                    "SHAVE ELF has multiple .arg.data sections");
      }
      arg_data = section;
      have_arg_data = true;
    }
  }
  if (!have_text || text.type != ELF_SECTION_PROGBITS || text.size == 0 || (text.flags & 6) != 6 ||
      text.address != SHAVE_TEXT_ADDRESS || entry != text.address) {
    return fail(diagnostic, NPUNLOCK_STATUS_UNSUPPORTED,
                "SHAVE .text or entry does not match the 3720xx ACT contract");
  }
  if (!have_arg_data || arg_data.address != SHAVE_ARG_ADDRESS || arg_data.size != 0) {
    return fail(diagnostic, NPUNLOCK_STATUS_UNSUPPORTED,
                "SHAVE .arg.data must be empty at 0x1e000000");
  }

  for (index = 0; index < program_count; ++index) {
    size_t offset;
    const uint8_t *program;
    uint32_t file_offset;
    uint32_t virtual_address;
    uint32_t file_size;
    uint32_t memory_size;
    uint32_t flags;
    size_t text_file_relative;
    size_t text_memory_relative;
    if (!npunlock_checked_add_size(program_table, (size_t)index * program_entry_size, &offset) ||
        !range_is_valid(elf.size, offset, ELF32_PROGRAM_SIZE)) {
      return fail(diagnostic, NPUNLOCK_STATUS_MALFORMED_INPUT,
                  "ELF32 program header is out of bounds");
    }
    program = elf.data + offset;
    file_offset = load_u32(program + 4);
    virtual_address = load_u32(program + 8);
    file_size = load_u32(program + 16);
    memory_size = load_u32(program + 20);
    flags = load_u32(program + 24);
    if (file_size > memory_size || !range_is_valid(elf.size, file_offset, file_size)) {
      return fail(diagnostic, NPUNLOCK_STATUS_MALFORMED_INPUT, "ELF32 program extent is invalid");
    }
    if (load_u32(program) == ELF_PROGRAM_LOAD && (flags & 5) == 5 && text.offset >= file_offset &&
        text.address >= virtual_address) {
      text_file_relative = (size_t)text.offset - file_offset;
      text_memory_relative = (size_t)text.address - virtual_address;
      if (range_is_valid(file_size, text_file_relative, text.size) &&
          range_is_valid(memory_size, text_memory_relative, text.size)) {
        text_is_loadable = true;
      }
    }
  }
  if (!text_is_loadable) {
    return fail(diagnostic, NPUNLOCK_STATUS_UNSUPPORTED,
                "SHAVE .text is not covered by an executable load segment");
  }

  image->file_offset = text.offset;
  image->size = text.size;
  image->address = text.address;
  image->entry = entry;
  return NPUNLOCK_STATUS_OK;
}
