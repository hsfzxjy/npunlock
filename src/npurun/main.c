#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "npunlock/graphinfer.h"
#include "npunlock/ir2blob.h"
#include "npunlock/patchblob.h"
#include "npunlock/shavecc.h"
#include "npunlock/version.h"

#include "internal.h"

#define MAX_TARGETS 32u
#define MAX_DEFINITIONS 32u
#define MOVITOOLS_DIRECTORY_ENV "NPUNLOCK_MOVITOOLS_DIR"

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
  const char *run_input_path;
  const char *run_output_path;
  char *owned_movi_directory;
  const char *definitions[MAX_DEFINITIONS];
  size_t definition_count;
  uint32_t invocations[MAX_TARGETS];
  size_t invocation_count;
  uint32_t ranges[MAX_TARGETS];
  size_t range_count;
  uint32_t patch_position;
  uint32_t input_count;
  uint64_t element_count;
  uint64_t span_bytes;
  uint32_t timeout_ms;
  uint32_t image_alignment;
  uint32_t tail_padding;
  uint32_t run_input_index;
  int run_add1;
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
  printf("         --shave-source KERNEL.c [--movi-dll-dir MVC_DEPEND] \\\n");
  printf("         [--linker-script FILE] \\\n");
  printf("         --patch-position N \\\n");
  printf("         --output GRAPH.blob --manifest BUILD.json [options]\n");
  printf("options: --build-flags TEXT --timeout-ms N --compiler-definition NAME=VALUE\n");
  printf("         --image-alignment N --tail-padding N --movi-worker FILE --ir-worker FILE\n");
  printf("         explicit override: --patch-invocation N --patch-range N [repeat both]\n");
  printf("         --input-count N --element-count N --span-bytes N\n");
  printf("         --run-add1 --run-input FILE --run-output FILE [--run-input-index N]\n");
  printf("environment: %s supplies the MVC_DEPEND root when --movi-dll-dir is omitted\n",
         MOVITOOLS_DIRECTORY_ENV);
}

static char *copy_environment_value(const char *name) {
#ifdef _WIN32
  char *value = NULL;
  size_t size = 0;
  if (_dupenv_s(&value, &size, name) != 0 || size <= 1) {
    free(value);
    return NULL;
  }
  return value;
#else
  const char *value = getenv(name);
  char *copy;
  size_t size;
  if (value == NULL || value[0] == 0) {
    return NULL;
  }
  size = strlen(value) + 1u;
  copy = (char *)malloc(size);
  if (copy != NULL) {
    memcpy(copy, value, size);
  }
  return copy;
#endif
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
  build->patch_position = PATCHBLOB_UNUSED_INDEX;
  build->input_count = 1;
  build->timeout_ms = 20000;
  build->image_alignment = 0x400;
  build->tail_padding = 0x80;
  build->run_input_index = 0;
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
    STRING_OPTION("--run-input", run_input_path)
    STRING_OPTION("--run-output", run_output_path)
#undef STRING_OPTION
    if (strcmp(option, "--run-add1") == 0) {
      build->run_add1 = 1;
    } else if (strcmp(option, "--patch-position") == 0) {
      if (build->patch_position != PATCHBLOB_UNUSED_INDEX ||
          !option_value(argument_count, arguments, &index, &value) ||
          !parse_u32(value, &build->patch_position) ||
          build->patch_position == PATCHBLOB_UNUSED_INDEX) {
        fprintf(stderr, "invalid --patch-position\n");
        return 0;
      }
    } else if (strcmp(option, "--patch-invocation") == 0) {
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
    } else if (strcmp(option, "--run-input-index") == 0) {
      if (!option_value(argument_count, arguments, &index, &value) ||
          !parse_u32(value, &build->run_input_index)) {
        return 0;
      }
    } else {
      fprintf(stderr, "unknown build option: %s\n", option);
      return 0;
    }
  }
  if (build->movi_directory == NULL) {
    build->owned_movi_directory = copy_environment_value(MOVITOOLS_DIRECTORY_ENV);
    build->movi_directory = build->owned_movi_directory;
  }
  if (build->movi_directory == NULL) {
    fprintf(stderr, "--movi-dll-dir or %s must supply the MVC_DEPEND root\n",
            MOVITOOLS_DIRECTORY_ENV);
    return 0;
  }
  if (build->ir_path == NULL || build->source_path == NULL || build->output_path == NULL ||
      build->manifest_path == NULL || build->timeout_ms == 0) {
    fprintf(stderr, "missing or inconsistent required build options\n");
    return 0;
  }
  if (build->patch_position != PATCHBLOB_UNUSED_INDEX) {
    if (build->invocation_count != 0 || build->range_count != 0) {
      fprintf(stderr, "--patch-position cannot be combined with explicit patch indices\n");
      return 0;
    }
  } else if (build->invocation_count == 0 || build->invocation_count != build->range_count ||
             build->input_count == 0 || build->element_count == 0 || build->span_bytes == 0) {
    fprintf(stderr, "explicit patch selection requires indices and a tensor contract\n");
    return 0;
  }
  if ((build->run_add1 && (build->run_input_path == NULL || build->run_output_path == NULL)) ||
      (!build->run_add1 && (build->run_input_path != NULL || build->run_output_path != NULL))) {
    fprintf(stderr, "--run-add1, --run-input, and --run-output must be specified together\n");
    return 0;
  }
  if (strcmp(build->output_path, build->manifest_path) == 0 ||
      strcmp(build->output_path, build->ir_path) == 0 ||
      strcmp(build->output_path, build->source_path) == 0 ||
      (build->linker_script_path != NULL &&
       strcmp(build->output_path, build->linker_script_path) == 0) ||
      (build->weights_path != NULL && strcmp(build->output_path, build->weights_path) == 0) ||
      strcmp(build->manifest_path, build->ir_path) == 0 ||
      strcmp(build->manifest_path, build->source_path) == 0 ||
      (build->linker_script_path != NULL &&
       strcmp(build->manifest_path, build->linker_script_path) == 0) ||
      (build->weights_path != NULL && strcmp(build->manifest_path, build->weights_path) == 0) ||
      (build->run_output_path != NULL &&
       (strcmp(build->run_output_path, build->output_path) == 0 ||
        strcmp(build->run_output_path, build->manifest_path) == 0 ||
        strcmp(build->run_output_path, build->ir_path) == 0 ||
        strcmp(build->run_output_path, build->source_path) == 0 ||
        (build->linker_script_path != NULL &&
         strcmp(build->run_output_path, build->linker_script_path) == 0) ||
        (build->weights_path != NULL &&
         strcmp(build->run_output_path, build->weights_path) == 0)))) {
    fprintf(stderr, "output and manifest paths must not name an input or each other\n");
    return 0;
  }
  if (build->run_input_path != NULL &&
      (strcmp(build->run_input_path, build->output_path) == 0 ||
       strcmp(build->run_input_path, build->manifest_path) == 0 ||
       strcmp(build->run_input_path, build->run_output_path) == 0)) {
    fprintf(stderr, "run input must not name an output file\n");
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

static void report_worker_streams(const npunlock_buffer *stdout_log,
                                  const npunlock_buffer *stderr_log, npunlock_status status) {
  if (status != NPUNLOCK_STATUS_OK) {
    if (stdout_log != NULL && stdout_log->size != 0) {
      fwrite(stdout_log->data, 1, stdout_log->size, stdout);
      fflush(stdout);
    }
    if (stderr_log != NULL && stderr_log->size != 0) {
      fwrite(stderr_log->data, 1, stderr_log->size, stderr);
      fflush(stderr);
    }
  } else if (stderr_log != NULL && stderr_log->size != 0) {
    fputs("warning: worker wrote to stderr:\n", stderr);
    fwrite(stderr_log->data, 1, stderr_log->size, stderr);
    if (stderr_log->data[stderr_log->size - 1] != '\n') {
      fputc('\n', stderr);
    }
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
  int length = snprintf(path, sizeof(path), "%s\\bin\\%s", directory, name);
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
                          const patchblob_result *patch_result, const build_provenance *provenance,
                          const graphinfer_result *run_result) {
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
              "\"linker_script_source\": \"%s\", \"linker_script_sha256\": \"%s\", "
              "\"movi_dll_directory\": ",
              shave_result->elf.size, provenance->source_hash,
              build->linker_script_path == NULL ? "builtin" : "caller",
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
      fprintf(stream, ",\n  \"execution\": ") < 0) {
    fclose(stream);
    return 0;
  }
  if (run_result == NULL) {
    if (fprintf(stream, "null\n}\n") < 0) {
      fclose(stream);
      return 0;
    }
  } else if (fprintf(stream,
                     "{\"oracle\": \"fp16_add1_exact\", \"driver_index\": %u, "
                     "\"device_index\": %u, \"driver_version\": %u, "
                     "\"device_vendor_id\": %u, \"device_id\": %u, "
                     "\"element_count\": %" PRIu64 ", \"mismatch_count\": 0}\n}\n",
                     run_result->selected_driver_index, run_result->selected_device_index,
                     run_result->driver_version, run_result->device_vendor_id,
                     run_result->device_id,
                     (uint64_t)(run_result->outputs[0].data.size / sizeof(uint16_t))) < 0) {
    fclose(stream);
    return 0;
  }
  if (fclose(stream) != 0) {
    return 0;
  }
  return 1;
}

static uint16_t float_to_fp16(float value) {
  uint32_t bits;
  uint16_t sign;
  uint32_t exponent_bits;
  uint32_t mantissa;
  int exponent;
  memcpy(&bits, &value, sizeof(bits));
  sign = (uint16_t)((bits >> 16) & 0x8000u);
  exponent_bits = (bits >> 23) & 0xffu;
  mantissa = bits & 0x7fffffu;
  if (exponent_bits == 0xffu) {
    uint16_t payload = (uint16_t)(mantissa >> 13);
    return mantissa == 0 ? (uint16_t)(sign | 0x7c00u)
                         : (uint16_t)(sign | 0x7c00u | (payload == 0 ? 1 : payload));
  }
  exponent = (int)exponent_bits - 127 + 15;
  if (exponent >= 31) {
    return (uint16_t)(sign | 0x7c00u);
  }
  if (exponent <= 0) {
    uint32_t rounded;
    uint32_t remainder;
    uint32_t halfway;
    unsigned shift;
    if (exponent < -10) {
      return sign;
    }
    mantissa |= 0x800000u;
    shift = (unsigned)(14 - exponent);
    rounded = mantissa >> shift;
    remainder = mantissa & ((1u << shift) - 1u);
    halfway = 1u << (shift - 1u);
    if (remainder > halfway || (remainder == halfway && (rounded & 1u) != 0)) {
      ++rounded;
    }
    return (uint16_t)(sign | rounded);
  }
  {
    uint32_t rounded = mantissa >> 13;
    uint32_t remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (rounded & 1u) != 0)) {
      ++rounded;
      if (rounded == 0x400u) {
        rounded = 0;
        ++exponent;
        if (exponent >= 31) {
          return (uint16_t)(sign | 0x7c00u);
        }
      }
    }
    return (uint16_t)(sign | ((uint16_t)exponent << 10) | rounded);
  }
}

static float fp16_to_float(uint16_t value) {
  uint32_t sign = (uint32_t)(value & 0x8000u) << 16;
  uint32_t exponent = (value >> 10) & 0x1fu;
  uint32_t mantissa = value & 0x03ffu;
  uint32_t bits;
  float result;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      int unbiased = -14;
      while ((mantissa & 0x0400u) == 0) {
        mantissa <<= 1;
        --unbiased;
      }
      mantissa &= 0x03ffu;
      bits = sign | ((uint32_t)(unbiased + 127) << 23) | (mantissa << 13);
    }
  } else if (exponent == 0x1fu) {
    bits = sign | 0x7f800000u | (mantissa << 13);
  } else {
    bits = sign | ((exponent - 15u + 127u) << 23) | (mantissa << 13);
  }
  memcpy(&result, &bits, sizeof(result));
  return result;
}

static int validate_add1_output(npunlock_view input, const graphinfer_result *result) {
  const graphinfer_output *output;
  const uint16_t *input_values = (const uint16_t *)input.data;
  const uint16_t *actual;
  uint64_t element_count;
  uint64_t index;
  uint64_t mismatches = 0;
  if (result->output_count != 1 || result->outputs == NULL ||
      result->outputs[0].precision != GRAPHINFER_PRECISION_FP16) {
    fprintf(stderr, "add1 execution requires exactly one FP16 graph output\n");
    return 0;
  }
  output = &result->outputs[0];
  if (input.size == 0 || input.size % sizeof(uint16_t) != 0 || output->data.size != input.size) {
    fprintf(stderr, "add1 input/output byte sizes do not match\n");
    return 0;
  }
  actual = (const uint16_t *)output->data.data;
  element_count = input.size / sizeof(uint16_t);
  for (index = 0; index < element_count; ++index) {
    uint16_t expected = float_to_fp16(fp16_to_float(input_values[index]) + 1.0f);
    if (actual[index] != expected) {
      if (mismatches < 8) {
        fprintf(stderr, "add1 mismatch at %" PRIu64 ": expected 0x%04x, got 0x%04x\n", index,
                expected, actual[index]);
      }
      ++mismatches;
    }
  }
  if (mismatches != 0) {
    fprintf(stderr, "add1 oracle failed: %" PRIu64 "/%" PRIu64 " mismatches\n", mismatches,
            element_count);
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
  file_buffer run_input = {0};
  npunlock_view definition_views[MAX_DEFINITIONS];
  patchblob_target targets[MAX_TARGETS];
  ir2blob_options ir_options = {0};
  shavecc_options shave_options = {0};
  patchblob_options patch_options = {0};
  ir2blob_result ir_result = {0};
  shavecc_result shave_result = {0};
  patchblob_result patch_result = {0};
  patchblob_discovery_result discovery_result = {0};
  graphinfer_result run_result = {0};
  build_provenance provenance = {0};
  npunlock_status status;
  size_t index;
  size_t selected_target_count = 0;
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
  if (build->linker_script_path != NULL) {
    linker_script = read_file(build->linker_script_path, 0);
  }
  if (source.size == SIZE_MAX || linker_script.size == SIZE_MAX) {
    goto cleanup;
  }
  if (build->run_add1) {
    run_input = read_file(build->run_input_path, 0);
    if (run_input.size == SIZE_MAX) {
      goto cleanup;
    }
  }
  hash_view((npunlock_view){ir.data, ir.size}, provenance.ir_hash);
  hash_view((npunlock_view){weights.data, weights.size}, provenance.weights_hash);
  hash_view((npunlock_view){source.data, source.size}, provenance.source_hash);
  hash_view(build->linker_script_path == NULL
                ? shavecc_default_linker_script()
                : (npunlock_view){linker_script.data, linker_script.size},
            provenance.linker_script_hash);
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
  report_worker_streams(&ir_result.stdout_log, &ir_result.stderr_log, status);
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
  report_worker_streams(&shave_result.stdout_log, &shave_result.stderr_log, status);
  if (status != NPUNLOCK_STATUS_OK) {
    print_diagnostic(&shave_result.diagnostic);
    goto cleanup;
  }
  if (build->patch_position != PATCHBLOB_UNUSED_INDEX) {
    status = patchblob_discover_targets(
        (npunlock_view){ir_result.graph_blob.data, ir_result.graph_blob.size}, &discovery_result);
    if (status != NPUNLOCK_STATUS_OK) {
      print_diagnostic(&discovery_result.diagnostic);
      goto cleanup;
    }
    if (build->patch_position >= discovery_result.group_count) {
      fprintf(stderr, "--patch-position %u is outside the %zu discovered ACT groups\n",
              build->patch_position, discovery_result.group_count);
      goto cleanup;
    }
    for (index = 0; index < discovery_result.target_count; ++index) {
      if (discovery_result.targets[index].group_index == build->patch_position) {
        if (selected_target_count == MAX_TARGETS) {
          fprintf(stderr, "selected ACT group exceeds the CLI target limit\n");
          goto cleanup;
        }
        targets[selected_target_count++] = discovery_result.targets[index].target;
      }
    }
  } else {
    selected_target_count = build->invocation_count;
    for (index = 0; index < selected_target_count; ++index) {
      targets[index].struct_size = sizeof(targets[index]);
      targets[index].invocation_index = build->invocations[index];
      targets[index].range_index = build->ranges[index];
      targets[index].expected_input_count = build->input_count;
      targets[index].expected_element_count = build->element_count;
      targets[index].expected_span_bytes = build->span_bytes;
      targets[index].required_contract_flags =
          PATCHBLOB_CONTRACT_STATIC | PATCHBLOB_CONTRACT_DENSE | PATCHBLOB_CONTRACT_FP16 |
          PATCHBLOB_CONTRACT_CMX | PATCHBLOB_CONTRACT_DISJOINT_OUTPUT;
    }
  }
  patch_options.struct_size = sizeof(patch_options);
  patch_options.image_alignment = build->image_alignment;
  patch_options.tail_padding = build->tail_padding;
  status = patchblob_patch(&patch_options,
                           (npunlock_view){ir_result.graph_blob.data, ir_result.graph_blob.size},
                           (npunlock_view){shave_result.elf.data, shave_result.elf.size}, targets,
                           selected_target_count, &patch_result);
  if (status != NPUNLOCK_STATUS_OK) {
    print_diagnostic(&patch_result.diagnostic);
    goto cleanup;
  }
  if (build->run_add1) {
    graphinfer_options infer_options = {0};
    graphinfer_input infer_input = {0};
    infer_options.struct_size = sizeof(infer_options);
    infer_options.driver_index = GRAPHINFER_AUTO_INDEX;
    infer_options.device_index = GRAPHINFER_AUTO_INDEX;
    infer_options.timeout_ms = build->timeout_ms;
    infer_input.struct_size = sizeof(infer_input);
    infer_input.argument_index = build->run_input_index;
    infer_input.data = (npunlock_view){run_input.data, run_input.size};
    status = graphinfer_infer(
        &infer_options, (npunlock_view){patch_result.graph_blob.data, patch_result.graph_blob.size},
        &infer_input, 1, &run_result);
    if (status != NPUNLOCK_STATUS_OK) {
      if (run_result.diagnostic.json.data != NULL) {
        fwrite(run_result.diagnostic.json.data, 1, run_result.diagnostic.json.size, stderr);
        fputc('\n', stderr);
      }
      goto cleanup;
    }
    if (!validate_add1_output((npunlock_view){run_input.data, run_input.size}, &run_result)) {
      goto cleanup;
    }
  }
  if (!write_file(build->output_path, patch_result.graph_blob.data, patch_result.graph_blob.size) ||
      !write_manifest(build->manifest_path, build, &ir_result, &shave_result, &patch_result,
                      &provenance, build->run_add1 ? &run_result : NULL) ||
      (build->run_add1 && !write_file(build->run_output_path, run_result.outputs[0].data.data,
                                      run_result.outputs[0].data.size))) {
    goto cleanup;
  }
  printf("wrote %zu-byte patched graph to %s\n", patch_result.graph_blob.size, build->output_path);
  printf("wrote build manifest to %s\n", build->manifest_path);
  if (build->run_add1) {
    printf("verified exact FP16 add1 semantics for %" PRIu64 " elements\n",
           (uint64_t)(run_result.outputs[0].data.size / sizeof(uint16_t)));
    printf("wrote %zu-byte raw output to %s\n", run_result.outputs[0].data.size,
           build->run_output_path);
  }
  success = 1;

cleanup:
  graphinfer_result_release(&run_result);
  patchblob_discovery_result_release(&discovery_result);
  patchblob_result_release(&patch_result);
  shavecc_result_release(&shave_result);
  ir2blob_result_release(&ir_result);
  free(linker_script.data);
  free(run_input.data);
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
    int result;
    if (!parse_build_arguments(argc, argv, &build)) {
      print_usage(argv[0]);
      free(build.owned_movi_directory);
      return 2;
    }
    result = run_build(&build);
    free(build.owned_movi_directory);
    return result;
  }
  print_usage(argv[0]);
  return 2;
}
