#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "npunlock/ir2blob.h"
#include "npunlock/patchblob.h"
#include "npunlock/shavecc.h"
#include "npunlock/version.h"

#include "internal.h"

#define MAX_TARGETS 32u
#define MAX_DEFINITIONS 32u

typedef struct file_buffer {
  uint8_t *data;
  size_t size;
} file_buffer;

typedef struct build_arguments {
  const char *ir_path;
  const char *weights_path;
  const char *source_path;
  const char *movi_directory;
  const char *linker_script_path;
  const char *output_path;
  const char *manifest_path;
  const char *build_flags;
  const char *movi_worker_path;
  const char *ir_worker_path;
  const char *definitions[MAX_DEFINITIONS];
  size_t definition_count;
  uint32_t invocations[MAX_TARGETS];
  size_t invocation_count;
  uint32_t ranges[MAX_TARGETS];
  size_t range_count;
  uint32_t input_count;
  uint64_t element_count;
  uint64_t span_bytes;
  uint32_t timeout_ms;
  uint32_t image_alignment;
  uint32_t tail_padding;
} build_arguments;

typedef struct build_provenance {
  char ir_hash[65];
  char weights_hash[65];
  char source_hash[65];
  char linker_script_hash[65];
  char movi_compile_hash[65];
  char movi_asm_hash[65];
  char movi_lld_hash[65];
} build_provenance;

static void print_usage(const char *program) {
  printf("usage: %s --version\n", program);
  printf("       %s build --ir MODEL.xml [--weights MODEL.bin] \\\n", program);
  printf("         --shave-source KERNEL.c --movi-dll-dir DIR --linker-script FILE \\\n");
  printf("         --patch-invocation N --patch-range N [repeat both options] \\\n");
  printf("         --input-count N --element-count N --span-bytes N \\\n");
  printf("         --output GRAPH.blob --manifest BUILD.json [options]\n");
  printf("options: --build-flags TEXT --timeout-ms N --compiler-definition NAME=VALUE\n");
  printf("         --image-alignment N --tail-padding N --movi-worker FILE --ir-worker FILE\n");
}

static int parse_u32(const char *text, uint32_t *result) {
  char *end = NULL;
  unsigned long long value;
  errno = 0;
  value = strtoull(text, &end, 0);
  if (errno != 0 || end == text || *end != 0 || value > UINT32_MAX) {
    return 0;
  }
  *result = (uint32_t)value;
  return 1;
}

static int parse_u64(const char *text, uint64_t *result) {
  char *end = NULL;
  unsigned long long value;
  errno = 0;
  value = strtoull(text, &end, 0);
  if (errno != 0 || end == text || *end != 0) {
    return 0;
  }
  *result = (uint64_t)value;
  return 1;
}

static int option_value(int argument_count, char **arguments, int *index, const char **value) {
  if (*index + 1 >= argument_count) {
    fprintf(stderr, "missing value for %s\n", arguments[*index]);
    return 0;
  }
  ++*index;
  *value = arguments[*index];
  return 1;
}

static int parse_build_arguments(int argument_count, char **arguments, build_arguments *build) {
  int index;
  memset(build, 0, sizeof(*build));
  build->build_flags = "";
  build->input_count = 1;
  build->timeout_ms = 20000;
  build->image_alignment = 0x400;
  build->tail_padding = 0x80;
  for (index = 2; index < argument_count; ++index) {
    const char *option = arguments[index];
    const char *value = NULL;
#define STRING_OPTION(name, field)                                                                 \
  if (strcmp(option, name) == 0) {                                                                 \
    if (!option_value(argument_count, arguments, &index, &value)) {                                \
      return 0;                                                                                    \
    }                                                                                              \
    build->field = value;                                                                          \
    continue;                                                                                      \
  }
    STRING_OPTION("--ir", ir_path)
    STRING_OPTION("--weights", weights_path)
    STRING_OPTION("--shave-source", source_path)
    STRING_OPTION("--movi-dll-dir", movi_directory)
    STRING_OPTION("--linker-script", linker_script_path)
    STRING_OPTION("--output", output_path)
    STRING_OPTION("--manifest", manifest_path)
    STRING_OPTION("--build-flags", build_flags)
    STRING_OPTION("--movi-worker", movi_worker_path)
    STRING_OPTION("--ir-worker", ir_worker_path)
#undef STRING_OPTION
    if (strcmp(option, "--patch-invocation") == 0) {
      if (build->invocation_count == MAX_TARGETS ||
          !option_value(argument_count, arguments, &index, &value) ||
          !parse_u32(value, &build->invocations[build->invocation_count])) {
        fprintf(stderr, "invalid --patch-invocation\n");
        return 0;
      }
      ++build->invocation_count;
    } else if (strcmp(option, "--patch-range") == 0) {
      if (build->range_count == MAX_TARGETS ||
          !option_value(argument_count, arguments, &index, &value) ||
          !parse_u32(value, &build->ranges[build->range_count])) {
        fprintf(stderr, "invalid --patch-range\n");
        return 0;
      }
      ++build->range_count;
    } else if (strcmp(option, "--compiler-definition") == 0) {
      if (build->definition_count == MAX_DEFINITIONS ||
          !option_value(argument_count, arguments, &index, &value)) {
        fprintf(stderr, "invalid --compiler-definition\n");
        return 0;
      }
      build->definitions[build->definition_count++] = value;
    } else if (strcmp(option, "--input-count") == 0) {
      if (!option_value(argument_count, arguments, &index, &value) ||
          !parse_u32(value, &build->input_count)) {
        return 0;
      }
    } else if (strcmp(option, "--element-count") == 0) {
      if (!option_value(argument_count, arguments, &index, &value) ||
          !parse_u64(value, &build->element_count)) {
        return 0;
      }
    } else if (strcmp(option, "--span-bytes") == 0) {
      if (!option_value(argument_count, arguments, &index, &value) ||
          !parse_u64(value, &build->span_bytes)) {
        return 0;
      }
    } else if (strcmp(option, "--timeout-ms") == 0) {
      if (!option_value(argument_count, arguments, &index, &value) ||
          !parse_u32(value, &build->timeout_ms)) {
        return 0;
      }
    } else if (strcmp(option, "--image-alignment") == 0) {
      if (!option_value(argument_count, arguments, &index, &value) ||
          !parse_u32(value, &build->image_alignment)) {
        return 0;
      }
    } else if (strcmp(option, "--tail-padding") == 0) {
      if (!option_value(argument_count, arguments, &index, &value) ||
          !parse_u32(value, &build->tail_padding)) {
        return 0;
      }
    } else {
      fprintf(stderr, "unknown build option: %s\n", option);
      return 0;
    }
  }
  if (build->ir_path == NULL || build->source_path == NULL || build->movi_directory == NULL ||
      build->linker_script_path == NULL || build->output_path == NULL ||
      build->manifest_path == NULL || build->invocation_count == 0 ||
      build->invocation_count != build->range_count || build->input_count == 0 ||
      build->element_count == 0 || build->span_bytes == 0 || build->timeout_ms == 0) {
    fprintf(stderr, "missing or inconsistent required build options\n");
    return 0;
  }
  if (strcmp(build->output_path, build->manifest_path) == 0 ||
      strcmp(build->output_path, build->ir_path) == 0 ||
      strcmp(build->output_path, build->source_path) == 0 ||
      strcmp(build->output_path, build->linker_script_path) == 0 ||
      (build->weights_path != NULL && strcmp(build->output_path, build->weights_path) == 0) ||
      strcmp(build->manifest_path, build->ir_path) == 0 ||
      strcmp(build->manifest_path, build->source_path) == 0 ||
      strcmp(build->manifest_path, build->linker_script_path) == 0 ||
      (build->weights_path != NULL && strcmp(build->manifest_path, build->weights_path) == 0)) {
    fprintf(stderr, "output and manifest paths must not name an input or each other\n");
    return 0;
  }
  return 1;
}

static file_buffer read_file(const char *path, int allow_empty) {
  file_buffer buffer = {NULL, SIZE_MAX};
  FILE *stream = NULL;
  long length;
#ifdef _WIN32
  if (fopen_s(&stream, path, "rb") != 0) {
    stream = NULL;
  }
#else
  stream = fopen(path, "rb");
#endif
  if (stream == NULL || fseek(stream, 0, SEEK_END) != 0) {
    if (stream != NULL) {
      fclose(stream);
    }
    fprintf(stderr, "could not open input file: %s\n", path);
    return buffer;
  }
  length = ftell(stream);
  if (length < 0 || (!allow_empty && length == 0) || fseek(stream, 0, SEEK_SET) != 0) {
    fclose(stream);
    fprintf(stderr, "invalid input file: %s\n", path);
    return buffer;
  }
  if (length != 0) {
    buffer.data = (uint8_t *)malloc((size_t)length);
    if (buffer.data == NULL || fread(buffer.data, 1, (size_t)length, stream) != (size_t)length) {
      free(buffer.data);
      buffer.data = NULL;
      fclose(stream);
      fprintf(stderr, "could not read input file: %s\n", path);
      return buffer;
    }
  }
  buffer.size = (size_t)length;
  fclose(stream);
  return buffer;
}

static int write_file(const char *path, const uint8_t *data, size_t size) {
  FILE *stream = NULL;
  int close_result;
#ifdef _WIN32
  if (fopen_s(&stream, path, "wb") != 0) {
    stream = NULL;
  }
#else
  stream = fopen(path, "wb");
#endif
  if (stream == NULL) {
    fprintf(stderr, "could not write output file: %s\n", path);
    return 0;
  }
  if (fwrite(data, 1, size, stream) != size) {
    fclose(stream);
    fprintf(stderr, "could not write output file: %s\n", path);
    return 0;
  }
  close_result = fclose(stream);
  if (close_result != 0) {
    fprintf(stderr, "could not finish output file: %s\n", path);
    return 0;
  }
  return 1;
}

static void print_diagnostic(const npunlock_diagnostic *diagnostic) {
  if (diagnostic != NULL && diagnostic->json.data != NULL) {
    fwrite(diagnostic->json.data, 1, diagnostic->json.size, stderr);
  }
}

static void hash_view(npunlock_view view, char text[65]) {
  static const char digits[] = "0123456789abcdef";
  uint8_t digest[32];
  size_t index;
  npunlock_sha256(view, digest);
  for (index = 0; index < sizeof(digest); ++index) {
    text[index * 2] = digits[digest[index] >> 4];
    text[index * 2 + 1] = digits[digest[index] & 15u];
  }
  text[64] = 0;
}

static int hash_movi_dll(const char *directory, const char *name, char hash[65]) {
  char path[4096];
  int length = snprintf(path, sizeof(path), "%s\\%s", directory, name);
  file_buffer file;
  if (length < 0 || (size_t)length >= sizeof(path)) {
    fprintf(stderr, "MoviTools DLL path is too long\n");
    return 0;
  }
  file = read_file(path, 0);
  if (file.size == SIZE_MAX) {
    return 0;
  }
  hash_view((npunlock_view){file.data, file.size}, hash);
  free(file.data);
  return 1;
}

static int write_json_string(FILE *stream, const char *value) {
  const unsigned char *cursor = (const unsigned char *)value;
  if (fputc('"', stream) == EOF) {
    return 0;
  }
  while (*cursor != 0) {
    if (*cursor == '"' || *cursor == '\\') {
      if (fputc('\\', stream) == EOF || fputc(*cursor, stream) == EOF) {
        return 0;
      }
    } else if (*cursor < 0x20) {
      if (fprintf(stream, "\\u%04x", *cursor) < 0) {
        return 0;
      }
    } else if (fputc(*cursor, stream) == EOF) {
      return 0;
    }
    ++cursor;
  }
  return fputc('"', stream) != EOF;
}

static int write_manifest(const char *path, const build_arguments *build,
                          const ir2blob_result *ir_result, const shavecc_result *shave_result,
                          const patchblob_result *patch_result,
                          const build_provenance *provenance) {
  FILE *stream = NULL;
  size_t index;
#ifdef _WIN32
  if (fopen_s(&stream, path, "wb") != 0) {
    stream = NULL;
  }
#else
  stream = fopen(path, "wb");
#endif
  if (stream == NULL) {
    return 0;
  }
  if (fprintf(stream, "{\n  \"schema\": \"npunlock.build.v1\",\n  \"inputs\": {\n    \"ir\": ") <
          0 ||
      !write_json_string(stream, build->ir_path) ||
      fprintf(stream, ", \"ir_sha256\": \"%s\",\n    \"weights\": ", provenance->ir_hash) < 0 ||
      !write_json_string(stream, build->weights_path == NULL ? "" : build->weights_path) ||
      fprintf(stream, ", \"weights_sha256\": \"%s\",\n    \"shave_source\": ",
              provenance->weights_hash) < 0 ||
      !write_json_string(stream, build->source_path) || fprintf(stream, "\n  },\n") < 0 ||
      fprintf(stream,
              "  \"shavecc\": {\"target\": \"3720xx\", \"entry\": "
              "\"controlled_act\", \"elf_size\": %zu, \"source_sha256\": \"%s\", "
              "\"linker_script_sha256\": \"%s\", \"movi_dll_directory\": ",
              shave_result->elf.size, provenance->source_hash,
              provenance->linker_script_hash) < 0 ||
      !write_json_string(stream, build->movi_directory) ||
      fprintf(stream,
              ", \"moviCompile64_sha256\": \"%s\", \"moviAsm64_sha256\": \"%s\", "
              "\"moviLLD64_sha256\": \"%s\"},\n",
              provenance->movi_compile_hash, provenance->movi_asm_hash,
              provenance->movi_lld_hash) < 0 ||
      fprintf(stream,
              "  \"ir2blob\": {\"driver_index\": %u, \"device_index\": %u, "
              "\"driver_version\": %u, \"device_vendor_id\": %u, \"device_id\": %u, "
              "\"graph_extension_version\": %u, \"compiler_version\": \"%u.%u\", "
              "\"max_opset\": %u, \"native_blob_size\": %zu, \"build_flags\": ",
              ir_result->selected_driver_index, ir_result->selected_device_index,
              ir_result->driver_version, ir_result->device_vendor_id, ir_result->device_id,
              ir_result->graph_extension_version, ir_result->compiler_version_major,
              ir_result->compiler_version_minor, ir_result->max_opset_version,
              ir_result->graph_blob.size) < 0 ||
      !write_json_string(stream, build->build_flags) || fprintf(stream, "},\n") < 0 ||
      fprintf(stream, "  \"compiler_definitions\": [") < 0) {
    fclose(stream);
    return 0;
  }
  for (index = 0; index < build->definition_count; ++index) {
    if ((index != 0 && fputc(',', stream) == EOF) ||
        !write_json_string(stream, build->definitions[index])) {
      fclose(stream);
      return 0;
    }
  }
  if (fprintf(stream, "],\n  \"patch\": ") < 0 ||
      fwrite(patch_result->report_json.data, 1, patch_result->report_json.size, stream) !=
          patch_result->report_json.size ||
      fprintf(stream, "}\n") < 0 || fclose(stream) != 0) {
    return 0;
  }
  return 1;
}

static int run_build(const build_arguments *build) {
  static const uint8_t cpu[] = "3720xx";
  static const uint8_t entry[] = "controlled_act";
  file_buffer ir = {0};
  file_buffer weights = {0};
  file_buffer source = {0};
  file_buffer linker_script = {0};
  npunlock_view definition_views[MAX_DEFINITIONS];
  patchblob_target targets[MAX_TARGETS];
  ir2blob_options ir_options = {0};
  shavecc_options shave_options = {0};
  patchblob_options patch_options = {0};
  ir2blob_result ir_result = {0};
  shavecc_result shave_result = {0};
  patchblob_result patch_result = {0};
  build_provenance provenance = {0};
  npunlock_status status;
  size_t index;
  int success = 0;

  ir = read_file(build->ir_path, 0);
  if (ir.size == SIZE_MAX) {
    goto cleanup;
  }
  if (build->weights_path != NULL) {
    weights = read_file(build->weights_path, 1);
    if (weights.size == SIZE_MAX) {
      goto cleanup;
    }
  }
  source = read_file(build->source_path, 0);
  linker_script = read_file(build->linker_script_path, 0);
  if (source.size == SIZE_MAX || linker_script.size == SIZE_MAX) {
    goto cleanup;
  }
  hash_view((npunlock_view){ir.data, ir.size}, provenance.ir_hash);
  hash_view((npunlock_view){weights.data, weights.size}, provenance.weights_hash);
  hash_view((npunlock_view){source.data, source.size}, provenance.source_hash);
  hash_view((npunlock_view){linker_script.data, linker_script.size}, provenance.linker_script_hash);
  if (!hash_movi_dll(build->movi_directory, "moviCompile64.dll", provenance.movi_compile_hash) ||
      !hash_movi_dll(build->movi_directory, "moviAsm64.dll", provenance.movi_asm_hash) ||
      !hash_movi_dll(build->movi_directory, "moviLLD64.dll", provenance.movi_lld_hash)) {
    goto cleanup;
  }
  ir_options.struct_size = sizeof(ir_options);
  ir_options.driver_index = IR2BLOB_AUTO_INDEX;
  ir_options.device_index = IR2BLOB_AUTO_INDEX;
  ir_options.timeout_ms = build->timeout_ms;
  ir_options.worker_executable_utf8 =
      (npunlock_view){(const uint8_t *)build->ir_worker_path,
                      build->ir_worker_path == NULL ? 0 : strlen(build->ir_worker_path)};
  ir_options.build_flags =
      (npunlock_view){(const uint8_t *)build->build_flags, strlen(build->build_flags)};
  status = ir2blob_compile(&ir_options, (npunlock_view){ir.data, ir.size},
                           (npunlock_view){weights.data, weights.size}, &ir_result);
  if (status != NPUNLOCK_STATUS_OK) {
    print_diagnostic(&ir_result.diagnostic);
    goto cleanup;
  }
  for (index = 0; index < build->definition_count; ++index) {
    definition_views[index] = (npunlock_view){(const uint8_t *)build->definitions[index],
                                              strlen(build->definitions[index])};
  }
  shave_options.struct_size = sizeof(shave_options);
  shave_options.movi_dll_directory_utf8 =
      (npunlock_view){(const uint8_t *)build->movi_directory, strlen(build->movi_directory)};
  shave_options.worker_executable_utf8 =
      (npunlock_view){(const uint8_t *)build->movi_worker_path,
                      build->movi_worker_path == NULL ? 0 : strlen(build->movi_worker_path)};
  shave_options.target_cpu = (npunlock_view){cpu, sizeof(cpu) - 1u};
  shave_options.entry_symbol = (npunlock_view){entry, sizeof(entry) - 1u};
  shave_options.compiler_definitions = definition_views;
  shave_options.compiler_definition_count = build->definition_count;
  shave_options.linker_script = (npunlock_view){linker_script.data, linker_script.size};
  shave_options.timeout_ms = build->timeout_ms;
  status =
      shavecc_compile(&shave_options, (npunlock_view){source.data, source.size}, &shave_result);
  if (status != NPUNLOCK_STATUS_OK) {
    print_diagnostic(&shave_result.diagnostic);
    goto cleanup;
  }
  for (index = 0; index < build->invocation_count; ++index) {
    targets[index].struct_size = sizeof(targets[index]);
    targets[index].invocation_index = build->invocations[index];
    targets[index].range_index = build->ranges[index];
    targets[index].expected_input_count = build->input_count;
    targets[index].expected_element_count = build->element_count;
    targets[index].expected_span_bytes = build->span_bytes;
    targets[index].required_contract_flags = PATCHBLOB_CONTRACT_STATIC | PATCHBLOB_CONTRACT_DENSE |
                                             PATCHBLOB_CONTRACT_FP16 | PATCHBLOB_CONTRACT_CMX |
                                             PATCHBLOB_CONTRACT_DISJOINT_OUTPUT;
  }
  patch_options.struct_size = sizeof(patch_options);
  patch_options.image_alignment = build->image_alignment;
  patch_options.tail_padding = build->tail_padding;
  status = patchblob_patch(&patch_options,
                           (npunlock_view){ir_result.graph_blob.data, ir_result.graph_blob.size},
                           (npunlock_view){shave_result.elf.data, shave_result.elf.size}, targets,
                           build->invocation_count, &patch_result);
  if (status != NPUNLOCK_STATUS_OK) {
    print_diagnostic(&patch_result.diagnostic);
    goto cleanup;
  }
  if (!write_file(build->output_path, patch_result.graph_blob.data, patch_result.graph_blob.size) ||
      !write_manifest(build->manifest_path, build, &ir_result, &shave_result, &patch_result,
                      &provenance)) {
    goto cleanup;
  }
  printf("wrote %zu-byte patched graph to %s\n", patch_result.graph_blob.size, build->output_path);
  printf("wrote build manifest to %s\n", build->manifest_path);
  success = 1;

cleanup:
  patchblob_result_release(&patch_result);
  shavecc_result_release(&shave_result);
  ir2blob_result_release(&ir_result);
  free(linker_script.data);
  free(source.data);
  free(weights.data);
  free(ir.data);
  return success ? 0 : 1;
}

int main(int argc, char **argv) {
  build_arguments build;
  if (argc == 2 && strcmp(argv[1], "--version") == 0) {
    printf("npunlock %s\n", NPUNLOCK_VERSION_STRING);
    return 0;
  }
  if (argc == 2 && (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0)) {
    print_usage(argv[0]);
    return 0;
  }
  if (argc >= 2 && strcmp(argv[1], "build") == 0) {
    if (!parse_build_arguments(argc, argv, &build)) {
      print_usage(argv[0]);
      return 2;
    }
    return run_build(&build);
  }
  print_usage(argv[0]);
  return 2;
}
