#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "internal.h"
#include "worker_process.h"

#define NPUNLOCK_WORKER_MAX_LOG_SIZE (4u * 1024u * 1024u)
#define NPUNLOCK_RESPONSE_FD 3

extern char **environ;

typedef struct byte_stream {
  uint8_t *data;
  size_t size;
  size_t capacity;
  size_t maximum;
  int fd;
  bool closed;
} byte_stream;

static bool view_has_nul(npunlock_view view) {
  return view.size != 0 && memchr(view.data, 0, view.size) != NULL;
}

static char *copy_view(npunlock_view view) {
  char *copy;
  if (!npunlock_view_is_valid(view) || view.size == 0 || view_has_nul(view) ||
      view.size == SIZE_MAX) {
    return NULL;
  }
  copy = (char *)malloc(view.size + 1);
  if (copy == NULL) {
    return NULL;
  }
  memcpy(copy, view.data, view.size);
  copy[view.size] = '\0';
  return copy;
}

static char *default_worker_path(void) {
  static const char worker_name[] = "npunlock_worker";
  char executable[4096];
  ssize_t length = readlink("/proc/self/exe", executable, sizeof(executable) - 1);
  char *separator;
  size_t directory_size;
  char *path;
  if (length <= 0 || (size_t)length >= sizeof(executable)) {
    return NULL;
  }
  executable[length] = '\0';
  separator = strrchr(executable, '/');
  if (separator == NULL) {
    return NULL;
  }
  directory_size = (size_t)(separator - executable) + 1;
  if (directory_size > SIZE_MAX - sizeof(worker_name)) {
    return NULL;
  }
  path = (char *)malloc(directory_size + sizeof(worker_name));
  if (path == NULL) {
    return NULL;
  }
  memcpy(path, executable, directory_size);
  memcpy(path + directory_size, worker_name, sizeof(worker_name));
  return path;
}

static bool relocate_fd(int *fd) {
  int replacement = fcntl(*fd, F_DUPFD_CLOEXEC, 10);
  if (replacement < 0) {
    return false;
  }
  close(*fd);
  *fd = replacement;
  return true;
}

static bool make_pipe(int ends[2]) {
  if (pipe(ends) != 0) {
    return false;
  }
  if (!relocate_fd(&ends[0]) || !relocate_fd(&ends[1])) {
    close(ends[0]);
    close(ends[1]);
    ends[0] = -1;
    ends[1] = -1;
    return false;
  }
  return true;
}

static bool make_request_socket(int ends[2]) {
  if (socketpair(AF_UNIX, SOCK_STREAM, 0, ends) != 0) {
    return false;
  }
  if (!relocate_fd(&ends[0]) || !relocate_fd(&ends[1])) {
    close(ends[0]);
    close(ends[1]);
    ends[0] = -1;
    ends[1] = -1;
    return false;
  }
  return true;
}

static void close_fd(int *fd) {
  if (*fd >= 0) {
    close(*fd);
    *fd = -1;
  }
}

static bool set_nonblocking(int fd) {
  int flags = fcntl(fd, F_GETFL, 0);
  return flags >= 0 && fcntl(fd, F_SETFL, flags | O_NONBLOCK) == 0;
}

static uint64_t monotonic_ms(void) {
  struct timespec now;
  if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
    return 0;
  }
  return (uint64_t)now.tv_sec * 1000u + (uint64_t)now.tv_nsec / 1000000u;
}

static npunlock_status reserve_stream(byte_stream *stream, size_t additional) {
  size_t required;
  size_t next;
  uint8_t *replacement;
  if (!npunlock_checked_add_size(stream->size, additional, &required) ||
      required > stream->maximum) {
    return NPUNLOCK_STATUS_OVERFLOW;
  }
  if (required <= stream->capacity) {
    return NPUNLOCK_STATUS_OK;
  }
  next = stream->capacity == 0 ? 4096 : stream->capacity;
  while (next < required) {
    if (next > stream->maximum / 2) {
      next = stream->maximum;
      break;
    }
    next *= 2;
  }
  replacement = (uint8_t *)realloc(stream->data, next);
  if (replacement == NULL) {
    return NPUNLOCK_STATUS_OUT_OF_MEMORY;
  }
  stream->data = replacement;
  stream->capacity = next;
  return NPUNLOCK_STATUS_OK;
}

static npunlock_status drain_stream(byte_stream *stream) {
  uint8_t chunk[16384];
  if (stream->closed) {
    return NPUNLOCK_STATUS_OK;
  }
  for (;;) {
    ssize_t received = read(stream->fd, chunk, sizeof(chunk));
    if (received > 0) {
      npunlock_status status = reserve_stream(stream, (size_t)received);
      if (status != NPUNLOCK_STATUS_OK) {
        return status;
      }
      memcpy(stream->data + stream->size, chunk, (size_t)received);
      stream->size += (size_t)received;
      continue;
    }
    if (received == 0) {
      stream->closed = true;
      close_fd(&stream->fd);
      return NPUNLOCK_STATUS_OK;
    }
    if (errno == EINTR) {
      continue;
    }
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
      return NPUNLOCK_STATUS_OK;
    }
    return NPUNLOCK_STATUS_IO_ERROR;
  }
}

static int add_child_actions(posix_spawn_file_actions_t *actions, const int request_pipe[2],
                             const int response_pipe[2], const int stdout_pipe[2],
                             const int stderr_pipe[2]) {
  const int descriptors[] = {request_pipe[0], request_pipe[1], response_pipe[0], response_pipe[1],
                             stdout_pipe[0],  stdout_pipe[1],  stderr_pipe[0],   stderr_pipe[1]};
  size_t index;
  if (posix_spawn_file_actions_adddup2(actions, request_pipe[0], STDIN_FILENO) != 0 ||
      posix_spawn_file_actions_adddup2(actions, response_pipe[1], NPUNLOCK_RESPONSE_FD) != 0 ||
      posix_spawn_file_actions_adddup2(actions, stdout_pipe[1], STDOUT_FILENO) != 0 ||
      posix_spawn_file_actions_adddup2(actions, stderr_pipe[1], STDERR_FILENO) != 0) {
    return 0;
  }
  for (index = 0; index < sizeof(descriptors) / sizeof(descriptors[0]); ++index) {
    if (posix_spawn_file_actions_addclose(actions, descriptors[index]) != 0) {
      return 0;
    }
  }
  return 1;
}

void npunlock_worker_process_result_release(npunlock_worker_process_result *result) {
  if (result == NULL) {
    return;
  }
  npunlock_buffer_release(&result->response);
  npunlock_buffer_release(&result->stdout_log);
  npunlock_buffer_release(&result->stderr_log);
  memset(result, 0, sizeof(*result));
}

npunlock_status npunlock_worker_process_run(npunlock_view worker_executable_utf8,
                                            const char *worker_mode, npunlock_view request,
                                            size_t maximum_response_size, uint32_t timeout_ms,
                                            npunlock_worker_process_result *result) {
  int request_pipe[2] = {-1, -1};
  int response_pipe[2] = {-1, -1};
  int stdout_pipe[2] = {-1, -1};
  int stderr_pipe[2] = {-1, -1};
  byte_stream response = {NULL, 0, 0, maximum_response_size, -1, false};
  byte_stream stdout_log = {NULL, 0, 0, NPUNLOCK_WORKER_MAX_LOG_SIZE, -1, false};
  byte_stream stderr_log = {NULL, 0, 0, NPUNLOCK_WORKER_MAX_LOG_SIZE, -1, false};
  posix_spawn_file_actions_t actions;
  posix_spawnattr_t attributes;
  bool actions_initialized = false;
  bool attributes_initialized = false;
  char response_fd_text[16];
  char *arguments[5];
  char *worker_path = NULL;
  pid_t process = -1;
  size_t request_offset = 0;
  uint64_t deadline;
  int wait_status = 0;
  bool process_exited = false;
  bool timed_out = false;
  npunlock_status status = NPUNLOCK_STATUS_INTERNAL_ERROR;

  if (result == NULL || worker_mode == NULL || worker_mode[0] == '\0' ||
      !npunlock_view_is_valid(worker_executable_utf8) || !npunlock_view_is_valid(request) ||
      request.size == 0 || maximum_response_size == 0 || timeout_ms == 0) {
    return NPUNLOCK_STATUS_INVALID_ARGUMENT;
  }
  memset(result, 0, sizeof(*result));
  worker_path =
      worker_executable_utf8.size == 0 ? default_worker_path() : copy_view(worker_executable_utf8);
  if (worker_path == NULL) {
    return worker_executable_utf8.size == 0 ? NPUNLOCK_STATUS_INTERNAL_ERROR
                                            : NPUNLOCK_STATUS_INVALID_ARGUMENT;
  }
  if (!make_request_socket(request_pipe) || !make_pipe(response_pipe) || !make_pipe(stdout_pipe) ||
      !make_pipe(stderr_pipe) || !set_nonblocking(request_pipe[1]) ||
      !set_nonblocking(response_pipe[0]) || !set_nonblocking(stdout_pipe[0]) ||
      !set_nonblocking(stderr_pipe[0])) {
    goto done;
  }
  if (posix_spawn_file_actions_init(&actions) != 0) {
    goto done;
  }
  actions_initialized = true;
  if (!add_child_actions(&actions, request_pipe, response_pipe, stdout_pipe, stderr_pipe) ||
      posix_spawnattr_init(&attributes) != 0) {
    goto done;
  }
  attributes_initialized = true;
  if (posix_spawnattr_setflags(&attributes, POSIX_SPAWN_SETPGROUP) != 0 ||
      posix_spawnattr_setpgroup(&attributes, 0) != 0) {
    goto done;
  }
  snprintf(response_fd_text, sizeof(response_fd_text), "%d", NPUNLOCK_RESPONSE_FD);
  arguments[0] = worker_path;
  arguments[1] = (char *)worker_mode;
  arguments[2] = "--response-fd";
  arguments[3] = response_fd_text;
  arguments[4] = NULL;
  {
    int spawn_error = posix_spawn(&process, worker_path, &actions, &attributes, arguments, environ);
    if (spawn_error != 0) {
      status = spawn_error == ENOENT ? NPUNLOCK_STATUS_NOT_FOUND : NPUNLOCK_STATUS_INTERNAL_ERROR;
      goto done;
    }
  }
  close_fd(&request_pipe[0]);
  close_fd(&response_pipe[1]);
  close_fd(&stdout_pipe[1]);
  close_fd(&stderr_pipe[1]);
  response.fd = response_pipe[0];
  response_pipe[0] = -1;
  stdout_log.fd = stdout_pipe[0];
  stdout_pipe[0] = -1;
  stderr_log.fd = stderr_pipe[0];
  stderr_pipe[0] = -1;
  deadline = monotonic_ms() + timeout_ms;
  status = NPUNLOCK_STATUS_OK;
  while (!process_exited || !response.closed || !stdout_log.closed || !stderr_log.closed) {
    struct pollfd poll_descriptors[4];
    byte_stream *streams[] = {&response, &stdout_log, &stderr_log};
    uint64_t now = monotonic_ms();
    int poll_timeout;
    int poll_result;
    size_t stream_index;
    pid_t wait_result;
    if (now >= deadline) {
      timed_out = true;
      status = NPUNLOCK_STATUS_TIMEOUT;
      break;
    }
    poll_timeout = (int)(deadline - now > 100 ? 100 : deadline - now);
    memset(poll_descriptors, 0, sizeof(poll_descriptors));
    poll_descriptors[0].fd = request_pipe[1];
    poll_descriptors[0].events = request_pipe[1] >= 0 ? POLLOUT : 0;
    for (stream_index = 0; stream_index < 3; ++stream_index) {
      poll_descriptors[stream_index + 1].fd = streams[stream_index]->fd;
      poll_descriptors[stream_index + 1].events = streams[stream_index]->closed ? 0 : POLLIN;
    }
    poll_result = poll(poll_descriptors, 4, poll_timeout);
    if (poll_result < 0 && errno != EINTR) {
      status = NPUNLOCK_STATUS_IO_ERROR;
      break;
    }
    if (request_pipe[1] >= 0 &&
        (poll_descriptors[0].revents & (POLLOUT | POLLHUP | POLLERR)) != 0) {
      ssize_t sent = send(request_pipe[1], request.data + request_offset,
                          request.size - request_offset, MSG_NOSIGNAL);
      if (sent > 0) {
        request_offset += (size_t)sent;
        if (request_offset == request.size) {
          close_fd(&request_pipe[1]);
        }
      } else if (sent < 0 && errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK &&
                 errno != EPIPE) {
        status = NPUNLOCK_STATUS_IO_ERROR;
        break;
      } else if (sent == 0 || errno == EPIPE) {
        close_fd(&request_pipe[1]);
      }
    }
    for (stream_index = 0; stream_index < 3; ++stream_index) {
      if (!streams[stream_index]->closed &&
          (poll_descriptors[stream_index + 1].revents & (POLLIN | POLLHUP | POLLERR)) != 0) {
        status = drain_stream(streams[stream_index]);
        if (status != NPUNLOCK_STATUS_OK) {
          break;
        }
      }
    }
    if (status != NPUNLOCK_STATUS_OK) {
      break;
    }
    wait_result = waitpid(process, &wait_status, WNOHANG);
    if (wait_result == process) {
      process_exited = true;
    } else if (wait_result < 0 && errno != EINTR) {
      status = NPUNLOCK_STATUS_INTERNAL_ERROR;
      break;
    }
  }
  if (status != NPUNLOCK_STATUS_OK || !process_exited) {
    kill(-process, SIGKILL);
    kill(process, SIGKILL);
    while (waitpid(process, &wait_status, 0) < 0 && errno == EINTR) {
    }
    process_exited = true;
  }
  close_fd(&request_pipe[1]);
  (void)drain_stream(&response);
  (void)drain_stream(&stdout_log);
  (void)drain_stream(&stderr_log);
  if (WIFEXITED(wait_status)) {
    result->process_exit_code = (uint32_t)WEXITSTATUS(wait_status);
  } else if (WIFSIGNALED(wait_status)) {
    result->process_exit_code = 128u + (uint32_t)WTERMSIG(wait_status);
  }
  if (status == NPUNLOCK_STATUS_OK && result->process_exit_code != 0) {
    status = NPUNLOCK_STATUS_DRIVER_FAILED;
  }
  if (timed_out) {
    status = NPUNLOCK_STATUS_TIMEOUT;
  }

done:
  close_fd(&request_pipe[0]);
  close_fd(&request_pipe[1]);
  close_fd(&response_pipe[0]);
  close_fd(&response_pipe[1]);
  close_fd(&stdout_pipe[0]);
  close_fd(&stdout_pipe[1]);
  close_fd(&stderr_pipe[0]);
  close_fd(&stderr_pipe[1]);
  close_fd(&response.fd);
  close_fd(&stdout_log.fd);
  close_fd(&stderr_log.fd);
  if (attributes_initialized) {
    posix_spawnattr_destroy(&attributes);
  }
  if (actions_initialized) {
    posix_spawn_file_actions_destroy(&actions);
  }
  free(worker_path);
  if (npunlock_buffer_adopt_malloc(response.data, response.size, &result->response) ==
      NPUNLOCK_STATUS_OK) {
    response.data = NULL;
  }
  if (npunlock_buffer_adopt_malloc(stdout_log.data, stdout_log.size, &result->stdout_log) ==
      NPUNLOCK_STATUS_OK) {
    stdout_log.data = NULL;
  }
  if (npunlock_buffer_adopt_malloc(stderr_log.data, stderr_log.size, &result->stderr_log) ==
      NPUNLOCK_STATUS_OK) {
    stderr_log.data = NULL;
  }
  free(response.data);
  free(stdout_log.data);
  free(stderr_log.data);
  return status;
}
