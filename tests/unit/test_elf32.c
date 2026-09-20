#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "elf32.h"

#define CHECK(expression)                                                                          \
  do {                                                                                             \
    if (!(expression)) {                                                                           \
      fprintf(stderr, "check failed at %s:%d: %s\n", __FILE__, __LINE__, #expression);             \
      return 1;                                                                                    \
    }                                                                                              \
  } while (0)

static void put16(uint8_t *data, size_t offset, uint16_t value) {
  data[offset] = (uint8_t)value;
  data[offset + 1] = (uint8_t)(value >> 8);
}

static void put32(uint8_t *data, size_t offset, uint32_t value) {
  data[offset] = (uint8_t)value;
  data[offset + 1] = (uint8_t)(value >> 8);
  data[offset + 2] = (uint8_t)(value >> 16);
  data[offset + 3] = (uint8_t)(value >> 24);
}

static void section(uint8_t *data, size_t table, unsigned index, uint32_t name, uint32_t type,
                    uint32_t flags, uint32_t address, uint32_t offset, uint32_t size,
                    uint32_t alignment) {
  size_t base = table + index * 40;
  put32(data, base, name);
  put32(data, base + 4, type);
  put32(data, base + 8, flags);
  put32(data, base + 12, address);
  put32(data, base + 16, offset);
  put32(data, base + 20, size);
  put32(data, base + 32, alignment);
}

static void make_elf(uint8_t *data, size_t size) {
  static const char names[] = "\0.arg.data\0.text\0.shstrtab\0";
  const size_t text_offset = 128;
  const size_t names_offset = 132;
  const size_t section_table = 192;
  memset(data, 0, size);
  memcpy(data,
         "\x7f"
         "ELF",
         4);
  data[4] = 1;
  data[5] = 1;
  data[6] = 1;
  put16(data, 16, 2);
  put16(data, 18, 2);
  put32(data, 20, 1);
  put32(data, 24, 0x1d000000);
  put32(data, 28, 52);
  put32(data, 32, (uint32_t)section_table);
  put16(data, 40, 52);
  put16(data, 42, 32);
  put16(data, 44, 1);
  put16(data, 46, 40);
  put16(data, 48, 4);
  put16(data, 50, 3);

  put32(data, 52, 1);
  put32(data, 56, (uint32_t)text_offset);
  put32(data, 60, 0x1d000000);
  put32(data, 64, 0x1d000000);
  put32(data, 68, 4);
  put32(data, 72, 4);
  put32(data, 76, 5);
  put32(data, 80, 16);

  data[text_offset] = 1;
  data[text_offset + 1] = 2;
  data[text_offset + 2] = 3;
  data[text_offset + 3] = 4;
  memcpy(data + names_offset, names, sizeof(names));
  section(data, section_table, 1, 1, 1, 2, 0x1e000000, (uint32_t)text_offset, 0, 1);
  section(data, section_table, 2, 11, 1, 6, 0x1d000000, (uint32_t)text_offset, 4, 16);
  section(data, section_table, 3, 17, 3, 0, 0, (uint32_t)names_offset, sizeof(names), 1);
}

int main(void) {
  uint8_t elf[352];
  uint8_t damaged[352];
  npunlock_shave_image image;
  npunlock_diagnostic diagnostic = {0};
  size_t length;

  make_elf(elf, sizeof(elf));
  CHECK(npunlock_parse_shave_elf((npunlock_view){elf, sizeof(elf)}, &image, &diagnostic) ==
        NPUNLOCK_STATUS_OK);
  CHECK(image.file_offset == 128 && image.size == 4 && image.address == 0x1d000000 &&
        image.entry == 0x1d000000);
  CHECK(diagnostic.json.data == NULL);

  for (length = 0; length < sizeof(elf); ++length) {
    CHECK(npunlock_parse_shave_elf((npunlock_view){elf, length}, &image, &diagnostic) !=
          NPUNLOCK_STATUS_OK);
    npunlock_diagnostic_release(&diagnostic);
  }

  memcpy(damaged, elf, sizeof(elf));
  put16(damaged, 18, 3);
  CHECK(npunlock_parse_shave_elf((npunlock_view){damaged, sizeof(damaged)}, &image, &diagnostic) ==
        NPUNLOCK_STATUS_UNSUPPORTED);
  npunlock_diagnostic_release(&diagnostic);

  memcpy(damaged, elf, sizeof(elf));
  put32(damaged, 192 + 40 + 20, 4);
  CHECK(npunlock_parse_shave_elf((npunlock_view){damaged, sizeof(damaged)}, &image, &diagnostic) ==
        NPUNLOCK_STATUS_UNSUPPORTED);
  npunlock_diagnostic_release(&diagnostic);

  memcpy(damaged, elf, sizeof(elf));
  put32(damaged, 56, 0);
  put32(damaged, 68, 4);
  CHECK(npunlock_parse_shave_elf((npunlock_view){damaged, sizeof(damaged)}, &image, &diagnostic) ==
        NPUNLOCK_STATUS_UNSUPPORTED);
  npunlock_diagnostic_release(&diagnostic);

  memcpy(damaged, elf, sizeof(elf));
  put32(damaged, 60, 0);
  put32(damaged, 72, 4);
  CHECK(npunlock_parse_shave_elf((npunlock_view){damaged, sizeof(damaged)}, &image, &diagnostic) ==
        NPUNLOCK_STATUS_UNSUPPORTED);
  npunlock_diagnostic_release(&diagnostic);
  return 0;
}
