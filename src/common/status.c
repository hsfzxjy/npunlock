#include "npunlock/error.h"

const char *npunlock_status_name(npunlock_status status) {
  switch (status) {
  case NPUNLOCK_STATUS_OK:
    return "ok";
  case NPUNLOCK_STATUS_INVALID_ARGUMENT:
    return "invalid_argument";
  case NPUNLOCK_STATUS_OUT_OF_MEMORY:
    return "out_of_memory";
  case NPUNLOCK_STATUS_OVERFLOW:
    return "overflow";
  case NPUNLOCK_STATUS_MALFORMED_INPUT:
    return "malformed_input";
  case NPUNLOCK_STATUS_UNSUPPORTED:
    return "unsupported";
  case NPUNLOCK_STATUS_NOT_FOUND:
    return "not_found";
  case NPUNLOCK_STATUS_TIMEOUT:
    return "timeout";
  case NPUNLOCK_STATUS_IO_ERROR:
    return "io_error";
  case NPUNLOCK_STATUS_TOOL_FAILED:
    return "tool_failed";
  case NPUNLOCK_STATUS_DRIVER_FAILED:
    return "driver_failed";
  case NPUNLOCK_STATUS_NOT_IMPLEMENTED:
    return "not_implemented";
  case NPUNLOCK_STATUS_INTERNAL_ERROR:
    return "internal_error";
  default:
    return "unknown";
  }
}
