#ifndef NPUNLOCK_BUFFER_H
#define NPUNLOCK_BUFFER_H

#include <stddef.h>
#include <stdint.h>

#include "npunlock/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct npunlock_view {
  const uint8_t *data;
  size_t size;
} npunlock_view;

typedef void (*npunlock_release_fn)(void *context, uint8_t *data, size_t size);

typedef struct npunlock_buffer {
  uint8_t *data;
  size_t size;
  npunlock_release_fn release;
  void *context;
} npunlock_buffer;

NPUNLOCK_COMMON_API void npunlock_buffer_release(npunlock_buffer *buffer);

#ifdef __cplusplus
}
#endif

#endif
