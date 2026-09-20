#include "internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static size_t escaped_size(const char *value) {
  size_t result = 0;
  const unsigned char *cursor = (const unsigned char *)value;
  while (*cursor != 0) {
    if (*cursor == '"' || *cursor == '\\') {
      result += 2;
    } else if (*cursor < 0x20) {
      result += 6;
    } else {
      ++result;
    }
    ++cursor;
  }
  return result;
}

static char *append_escaped(char *output, const char *value) {
  static const char hex[] = "0123456789abcdef";
  const unsigned char *cursor = (const unsigned char *)value;
  while (*cursor != 0) {
    if (*cursor == '"' || *cursor == '\\') {
      *output++ = '\\';
      *output++ = (char)*cursor;
    } else if (*cursor < 0x20) {
      *output++ = '\\';
      *output++ = 'u';
      *output++ = '0';
      *output++ = '0';
      *output++ = hex[*cursor >> 4];
      *output++ = hex[*cursor & 0x0f];
    } else {
      *output++ = (char)*cursor;
    }
    ++cursor;
  }
  return output;
}

npunlock_status npunlock_set_diagnostic(npunlock_diagnostic *diagnostic, npunlock_status status,
                                        const char *stage, const char *message) {
  static const char prefix[] = "{\"status\":\"";
  static const char middle_stage[] = "\",\"stage\":\"";
  static const char middle_message[] = "\",\"message\":\"";
  static const char suffix[] = "\"}\n";
  const char *status_name;
  size_t size;
  char *json;
  char *cursor;

  if (diagnostic == NULL || stage == NULL || message == NULL) {
    return NPUNLOCK_STATUS_INVALID_ARGUMENT;
  }
  npunlock_diagnostic_release(diagnostic);
  diagnostic->struct_size = (uint32_t)sizeof(*diagnostic);
  status_name = npunlock_status_name(status);
  size = sizeof(prefix) - 1 + escaped_size(status_name) + sizeof(middle_stage) - 1 +
         escaped_size(stage) + sizeof(middle_message) - 1 + escaped_size(message) + sizeof(suffix) -
         1;
  json = (char *)malloc(size + 1);
  if (json == NULL) {
    return NPUNLOCK_STATUS_OUT_OF_MEMORY;
  }
  cursor = json;
  memcpy(cursor, prefix, sizeof(prefix) - 1);
  cursor += sizeof(prefix) - 1;
  cursor = append_escaped(cursor, status_name);
  memcpy(cursor, middle_stage, sizeof(middle_stage) - 1);
  cursor += sizeof(middle_stage) - 1;
  cursor = append_escaped(cursor, stage);
  memcpy(cursor, middle_message, sizeof(middle_message) - 1);
  cursor += sizeof(middle_message) - 1;
  cursor = append_escaped(cursor, message);
  memcpy(cursor, suffix, sizeof(suffix) - 1);
  cursor += sizeof(suffix) - 1;
  *cursor = '\0';
  if (npunlock_buffer_adopt_malloc((uint8_t *)json, size, &diagnostic->json) !=
      NPUNLOCK_STATUS_OK) {
    free(json);
    return NPUNLOCK_STATUS_INTERNAL_ERROR;
  }
  return status;
}

void npunlock_diagnostic_release(npunlock_diagnostic *diagnostic) {
  if (diagnostic == NULL) {
    return;
  }
  npunlock_buffer_release(&diagnostic->json);
  diagnostic->struct_size = 0;
}
