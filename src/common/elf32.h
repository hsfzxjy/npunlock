#ifndef NPUNLOCK_COMMON_ELF32_H
#define NPUNLOCK_COMMON_ELF32_H

#include <stddef.h>
#include <stdint.h>

#include "npunlock/buffer.h"
#include "npunlock/error.h"

typedef struct npunlock_shave_image {
  size_t file_offset;
  size_t size;
  uint32_t address;
  uint32_t entry;
} npunlock_shave_image;

NPUNLOCK_COMMON_API npunlock_status npunlock_parse_shave_elf(npunlock_view elf,
                                                             npunlock_shave_image *image,
                                                             npunlock_diagnostic *diagnostic);

#endif
