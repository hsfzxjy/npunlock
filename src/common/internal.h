#ifndef NPUNLOCK_COMMON_INTERNAL_H
#define NPUNLOCK_COMMON_INTERNAL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "npunlock/buffer.h"
#include "npunlock/error.h"

NPUNLOCK_COMMON_API bool npunlock_view_is_valid(npunlock_view view);
NPUNLOCK_COMMON_API bool npunlock_checked_add_size(size_t left, size_t right, size_t *result);
NPUNLOCK_COMMON_API bool npunlock_checked_mul_size(size_t left, size_t right, size_t *result);
NPUNLOCK_COMMON_API npunlock_status npunlock_buffer_copy(npunlock_view source,
                                                         npunlock_buffer *destination);
NPUNLOCK_COMMON_API npunlock_status npunlock_buffer_adopt_malloc(uint8_t *data, size_t size,
                                                                 npunlock_buffer *destination);
NPUNLOCK_COMMON_API npunlock_status npunlock_set_diagnostic(npunlock_diagnostic *diagnostic,
                                                            npunlock_status status,
                                                            const char *stage, const char *message);

#endif
