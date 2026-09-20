#ifndef NPUNLOCK_ERROR_H
#define NPUNLOCK_ERROR_H

#include <stdint.h>

#include "npunlock/buffer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum npunlock_status {
  NPUNLOCK_STATUS_OK = 0,
  NPUNLOCK_STATUS_INVALID_ARGUMENT = 1,
  NPUNLOCK_STATUS_OUT_OF_MEMORY = 2,
  NPUNLOCK_STATUS_OVERFLOW = 3,
  NPUNLOCK_STATUS_MALFORMED_INPUT = 4,
  NPUNLOCK_STATUS_UNSUPPORTED = 5,
  NPUNLOCK_STATUS_NOT_FOUND = 6,
  NPUNLOCK_STATUS_TIMEOUT = 7,
  NPUNLOCK_STATUS_IO_ERROR = 8,
  NPUNLOCK_STATUS_TOOL_FAILED = 9,
  NPUNLOCK_STATUS_DRIVER_FAILED = 10,
  NPUNLOCK_STATUS_NOT_IMPLEMENTED = 11,
  NPUNLOCK_STATUS_INTERNAL_ERROR = 12
} npunlock_status;

typedef struct npunlock_diagnostic {
  uint32_t struct_size;
  npunlock_buffer json;
} npunlock_diagnostic;

NPUNLOCK_COMMON_API const char *npunlock_status_name(npunlock_status status);
NPUNLOCK_COMMON_API void npunlock_diagnostic_release(npunlock_diagnostic *diagnostic);

#ifdef __cplusplus
}
#endif

#endif
