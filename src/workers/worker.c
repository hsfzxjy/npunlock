#include <string.h>

#include "worker_entry.h"

int main(int argc, char **argv) {
  if (argc < 2) {
    return 2;
  }
  if (strcmp(argv[1], "movi") == 0) {
    return npunlock_movi_worker_main(argc - 1, argv + 1);
  }
  if (strcmp(argv[1], "ir") == 0) {
    return npunlock_ir_worker_main(argc - 1, argv + 1);
  }
  return 2;
}
