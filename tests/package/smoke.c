#include <stdio.h>

#include "npunlock/graphinfer.h"
#include "npunlock/shavecc.h"
#include "npunlock/version.h"

int main(void) {
  graphinfer_result result = {0};
  npunlock_view linker_script = shavecc_default_linker_script();
  result.struct_size = sizeof(result);
  graphinfer_result_release(&result);
  if (linker_script.data == NULL || linker_script.size == 0) {
    return 1;
  }
  printf("npunlock %s: built-in linker script is %zu bytes\n", NPUNLOCK_VERSION_STRING,
         linker_script.size);
  return 0;
}
