#include <stdio.h>

#include "npunlock/error.h"
#include "npunlock/version.h"

int main(void) {
  printf("npunlock %s: %s\n", NPUNLOCK_VERSION_STRING, npunlock_status_name(NPUNLOCK_STATUS_OK));
  return 0;
}
