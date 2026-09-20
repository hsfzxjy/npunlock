#include "npunlock/buffer.h"

#include <stdlib.h>
#include <string.h>

#include "internal.h"

static void release_malloc(void *context, uint8_t *data, size_t size) {
  (void)context;
  (void)size;
  free(data);
}

bool npunlock_view_is_valid(npunlock_view view) { return view.data != NULL || view.size == 0; }

void npunlock_buffer_release(npunlock_buffer *buffer) {
  if (buffer == NULL) {
    return;
  }
  if (buffer->release != NULL && buffer->data != NULL) {
    buffer->release(buffer->context, buffer->data, buffer->size);
  }
  memset(buffer, 0, sizeof(*buffer));
}

npunlock_status npunlock_buffer_copy(npunlock_view source, npunlock_buffer *destination) {
  uint8_t *copy;

  if (destination == NULL || !npunlock_view_is_valid(source)) {
    return NPUNLOCK_STATUS_INVALID_ARGUMENT;
  }
  memset(destination, 0, sizeof(*destination));
  if (source.size == 0) {
    return NPUNLOCK_STATUS_OK;
  }
  copy = (uint8_t *)malloc(source.size);
  if (copy == NULL) {
    return NPUNLOCK_STATUS_OUT_OF_MEMORY;
  }
  memcpy(copy, source.data, source.size);
  destination->data = copy;
  destination->size = source.size;
  destination->release = release_malloc;
  return NPUNLOCK_STATUS_OK;
}

npunlock_status npunlock_buffer_adopt_malloc(uint8_t *data, size_t size,
                                             npunlock_buffer *destination) {
  if (destination == NULL || (data == NULL && size != 0)) {
    return NPUNLOCK_STATUS_INVALID_ARGUMENT;
  }
  memset(destination, 0, sizeof(*destination));
  destination->data = data;
  destination->size = size;
  destination->release = data != NULL ? release_malloc : NULL;
  return NPUNLOCK_STATUS_OK;
}
