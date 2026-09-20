#include <stdio.h>

#include "npunlock/error.h"
#include "npunlock/graphinfer.h"
#include "npunlock/version.h"

int main(void) {
  graphinfer_result result = {0};
  result.struct_size = sizeof(result);
  graphinfer_result_release(&result);
  printf("npunlock %s: %s\n", NPUNLOCK_VERSION_STRING, npunlock_status_name(NPUNLOCK_STATUS_OK));
  return 0;
}
