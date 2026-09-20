#include <stdio.h>
#include <string.h>

#include "npunlock/version.h"

static void print_usage(const char *program) {
  printf("usage: %s --version\n", program);
  printf("       %s build [options]  (implementation pending)\n", program);
}

int main(int argc, char **argv) {
  if (argc == 2 && strcmp(argv[1], "--version") == 0) {
    printf("npunlock %s\n", NPUNLOCK_VERSION_STRING);
    return 0;
  }
  if (argc == 2 && (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0)) {
    print_usage(argv[0]);
    return 0;
  }
  print_usage(argv[0]);
  return 2;
}
