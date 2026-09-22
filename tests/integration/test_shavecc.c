#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "npunlock/shavecc.h"

#define FIXTURE_ROOT "tests/fixtures/npu3720/"
#define LINKER_SCRIPT_PATH "src/shavecc/shave_kernel.ld"
#define KERNEL_HEADER_PATH "include/npunlock/npu3720_kernel.h"

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

int main(void) {
  static const uint8_t included_source[] =
      "#define MLIBM_DEFINE_LINK_COMPAT 1\n"
      "#include <npunlock/npu3720_kernel.h>\n"
      "void controlled_act(unsigned layerParams) {\n"
      "  act_abi_invocation invocation;\n"
      "  ACT_ABI_LOAD_INVOCATION32_OR_RETURN(layerParams, invocation);\n"
      "  const __fp16 *in = ACT_ABI_INPUT_PTR32(const __fp16, invocation, 0u);\n"
      "  __fp16 *out = ACT_ABI_OUTPUT_PTR32(__fp16, invocation, 1u);\n"
      "  for (unsigned i = 0; i < invocation.element_count; ++i) out[i] = in[i] + 1.0f;\n"
      "}\n";
  static const uint8_t target[] = "3720xx";
  static const uint8_t entry[] = "controlled_act";
  file_buffer source = {0};
  file_buffer script = {0};
  file_buffer kernel_header = {0};
  file_buffer expected = {0};
  shavecc_options options = {0};
  shavecc_result result = {0};
  npunlock_status status;
  char *movi_dll_directory = NULL;
  size_t movi_dll_directory_size = 0;
  int return_code = 1;

  if (_dupenv_s(&movi_dll_directory, &movi_dll_directory_size, "NPUNLOCK_MOVITOOLS_DIR") != 0 ||
      movi_dll_directory_size <= 1 || !read_file(FIXTURE_ROOT "add1-fp16.c", &source) ||
      !read_file(LINKER_SCRIPT_PATH, &script) || !read_file(KERNEL_HEADER_PATH, &kernel_header) ||
      !read_file(FIXTURE_ROOT "add1-fp16.elf", &expected)) {
    fprintf(stderr, "NPUNLOCK_MOVITOOLS_DIR must name the MVC_DEPEND root\n");
    goto done;
  }
  options.struct_size = sizeof(options);
  options.movi_dll_directory_utf8 =
      (npunlock_view){(const uint8_t *)movi_dll_directory, strlen(movi_dll_directory)};
  options.target_cpu = (npunlock_view){target, sizeof(target) - 1};
  options.entry_symbol = (npunlock_view){entry, sizeof(entry) - 1};
  options.timeout_ms = 8000;
  if (shavecc_default_linker_script().size != script.size ||
      memcmp(shavecc_default_linker_script().data, script.data, script.size) != 0) {
    fprintf(stderr, "embedded linker script differs from its vendored source\n");
    goto done;
  }
  if (shavecc_npu3720_kernel_header().size != kernel_header.size ||
      memcmp(shavecc_npu3720_kernel_header().data, kernel_header.data, kernel_header.size) != 0) {
    fprintf(stderr, "embedded NPU3720 kernel header differs from its installed source\n");
    goto done;
  }
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
  shavecc_result_release(&result);
  options.linker_script = (npunlock_view){script.data, script.size};
  status = shavecc_compile(&options, (npunlock_view){source.data, source.size}, &result);
  if (status != NPUNLOCK_STATUS_OK || result.elf.size != expected.size ||
      memcmp(result.elf.data, expected.data, expected.size) != 0) {
    fprintf(stderr, "caller linker-script override did not reproduce the retained ELF\n");
    goto done;
  }
  shavecc_result_release(&result);
  options.linker_script = (npunlock_view){0};
  status = shavecc_compile(&options, (npunlock_view){included_source, sizeof(included_source) - 1},
                           &result);
  if (status != NPUNLOCK_STATUS_OK || result.elf.size == 0) {
    fprintf(stderr, "bundled NPU3720 kernel include did not compile\n");
    if (result.diagnostic.json.data != NULL) {
      fwrite(result.diagnostic.json.data, 1, result.diagnostic.json.size, stderr);
    }
    goto done;
  }
  printf("validated linker scripts and bundled %zu-byte NPU3720 kernel header\n",
         kernel_header.size);
  return_code = 0;

done:
  shavecc_result_release(&result);
  release_file(&expected);
  release_file(&kernel_header);
  release_file(&script);
  release_file(&source);
  free(movi_dll_directory);
  return return_code;
}
