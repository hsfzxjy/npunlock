#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "npunlock/shavecc.h"

#define FIXTURE_ROOT "tests/fixtures/npu3720/"

typedef struct file_buffer {
  uint8_t *data;
  size_t size;
} file_buffer;

static int read_file(const char *path, file_buffer *buffer) {
  FILE *file = NULL;
  __int64 length;
  size_t received;
  memset(buffer, 0, sizeof(*buffer));
  if (fopen_s(&file, path, "rb") != 0 || file == NULL || _fseeki64(file, 0, SEEK_END) != 0) {
    if (file != NULL) {
      fclose(file);
    }
    return 0;
  }
  length = _ftelli64(file);
  if (length < 0 || (uint64_t)length > SIZE_MAX || length > 64 * 1024 * 1024 ||
      _fseeki64(file, 0, SEEK_SET) != 0) {
    fclose(file);
    return 0;
  }
  if (length != 0) {
    buffer->data = (uint8_t *)malloc((size_t)length);
    if (buffer->data == NULL) {
      fclose(file);
      return 0;
    }
  }
  received = length == 0 ? 0 : fread(buffer->data, 1, (size_t)length, file);
  fclose(file);
  if (received != (size_t)length) {
    free(buffer->data);
    memset(buffer, 0, sizeof(*buffer));
    return 0;
  }
  buffer->size = (size_t)length;
  return 1;
}

static void release_file(file_buffer *buffer) {
  free(buffer->data);
  memset(buffer, 0, sizeof(*buffer));
}

int main(int argc, char **argv) {
  static const uint8_t target[] = "3720xx";
  static const uint8_t entry[] = "controlled_act";
  file_buffer source = {0};
  file_buffer script = {0};
  file_buffer expected = {0};
  shavecc_options options = {0};
  shavecc_result result = {0};
  npunlock_status status;
  int return_code = 1;

  if (argc != 2 || !read_file(FIXTURE_ROOT "add1-fp16.c", &source) ||
      !read_file(FIXTURE_ROOT "shave_kernel.ld", &script) ||
      !read_file(FIXTURE_ROOT "add1-fp16.elf", &expected)) {
    fprintf(stderr, "usage: %s MOVI_DLL_DIR\n", argv[0]);
    goto done;
  }
  options.struct_size = sizeof(options);
  options.movi_dll_directory_utf8 = (npunlock_view){(const uint8_t *)argv[1], strlen(argv[1])};
  options.target_cpu = (npunlock_view){target, sizeof(target) - 1};
  options.entry_symbol = (npunlock_view){entry, sizeof(entry) - 1};
  options.linker_script = (npunlock_view){script.data, script.size};
  options.timeout_ms = 8000;
  status = shavecc_compile(&options, (npunlock_view){source.data, source.size}, &result);
  if (status != NPUNLOCK_STATUS_OK) {
    fprintf(stderr, "shavecc_compile failed: %s\n", npunlock_status_name(status));
    if (result.diagnostic.json.data != NULL) {
      fwrite(result.diagnostic.json.data, 1, result.diagnostic.json.size, stderr);
    }
    goto done;
  }
  if (result.elf.size != expected.size ||
      memcmp(result.elf.data, expected.data, expected.size) != 0) {
    fprintf(stderr,
            "linked ELF differs from the retained repeatable build (%zu versus %zu bytes)\n",
            result.elf.size, expected.size);
    goto done;
  }
  printf("validated linked SHAVE ELF: %zu bytes; retained output is byte-identical\n",
         result.elf.size);
  return_code = 0;

done:
  shavecc_result_release(&result);
  release_file(&expected);
  release_file(&script);
  release_file(&source);
  return return_code;
}
