#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static int parse_response_fd(int argc, char **argv) {
  char *end = NULL;
  long value;
  if (argc != 4 || strcmp(argv[2], "--response-fd") != 0) {
    return -1;
  }
  errno = 0;
  value = strtol(argv[3], &end, 10);
  if (errno != 0 || end == argv[3] || *end != '\0' || value < 0 || value > INT32_MAX) {
    return -1;
  }
  return (int)value;
}

static int write_all(int fd, const uint8_t *data, size_t size) {
  while (size != 0) {
    ssize_t written = write(fd, data, size);
    if (written > 0) {
      data += written;
      size -= (size_t)written;
    } else if (written < 0 && errno == EINTR) {
      continue;
    } else {
      return 0;
    }
  }
  return 1;
}

int main(int argc, char **argv) {
  uint8_t request[64];
  size_t used = 0;
  int response_fd = parse_response_fd(argc, argv);
  if (response_fd < 0) {
    return 2;
  }
  while (used != sizeof(request)) {
    ssize_t received = read(STDIN_FILENO, request + used, sizeof(request) - used);
    if (received > 0) {
      used += (size_t)received;
    } else if (received == 0) {
      break;
    } else if (errno != EINTR) {
      return 3;
    }
  }
  if (strcmp(argv[1], "echo") == 0) {
    static const uint8_t stdout_text[] = "fixture stdout\n";
    static const uint8_t stderr_text[] = "fixture stderr\n";
    if (!write_all(STDOUT_FILENO, stdout_text, sizeof(stdout_text) - 1) ||
        !write_all(STDERR_FILENO, stderr_text, sizeof(stderr_text) - 1) ||
        !write_all(response_fd, request, used)) {
      return 4;
    }
    return 0;
  }
  if (strcmp(argv[1], "hang") == 0) {
    struct timespec delay = {10, 0};
    pid_t descendant = fork();
    if (descendant < 0) {
      return 5;
    }
    while (nanosleep(&delay, &delay) != 0 && errno == EINTR) {
    }
    if (descendant > 0) {
      while (waitpid(descendant, NULL, 0) < 0 && errno == EINTR) {
      }
    }
    return 0;
  }
  return 2;
}
