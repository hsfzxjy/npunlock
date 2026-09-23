#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "npunlock/version.h"

#include "infer_engine.h"

typedef struct file_buffer {
  uint8_t *data;
  size_t size;
} file_buffer;

static void print_usage(const char *program) {
  printf("usage: %s --bundle ROOT [--report REPORT.json]\n", program);
  printf("       %s --version\n", program);
}

static char *join_path(const char *root, const char *name) {
  size_t root_size = strlen(root);
  size_t name_size = strlen(name);
  int needs_separator = root_size != 0 && root[root_size - 1] != '/';
  char *path;
  if (root_size > SIZE_MAX - name_size - (size_t)needs_separator - 1) {
    return NULL;
  }
  path = (char *)malloc(root_size + (size_t)needs_separator + name_size + 1);
  if (path == NULL) {
    return NULL;
  }
  memcpy(path, root, root_size);
  if (needs_separator) {
    path[root_size++] = '/';
  }
  memcpy(path + root_size, name, name_size + 1);
  return path;
}

static file_buffer read_file(const char *path, size_t maximum) {
  file_buffer buffer = {NULL, SIZE_MAX};
  FILE *stream = fopen(path, "rb");
  long length;
  if (stream == NULL || fseek(stream, 0, SEEK_END) != 0) {
    if (stream != NULL) {
      fclose(stream);
    }
    fprintf(stderr, "could not open %s: %s\n", path, strerror(errno));
    return buffer;
  }
  length = ftell(stream);
  if (length <= 0 || (uint64_t)length > SIZE_MAX || (uint64_t)length > maximum ||
      fseek(stream, 0, SEEK_SET) != 0) {
    fclose(stream);
    fprintf(stderr, "file is empty, unreadable, or exceeds the supported bound: %s\n", path);
    return buffer;
  }
  buffer.data = (uint8_t *)malloc((size_t)length);
  if (buffer.data == NULL || fread(buffer.data, 1, (size_t)length, stream) != (size_t)length) {
    free(buffer.data);
    buffer.data = NULL;
    fclose(stream);
    fprintf(stderr, "could not read %s\n", path);
    return buffer;
  }
  buffer.size = (size_t)length;
  fclose(stream);
  return buffer;
}

static int file_exists(const char *path) {
  FILE *stream = fopen(path, "rb");
  if (stream == NULL) {
    return 0;
  }
  fclose(stream);
  return 1;
}

static int write_json_string(FILE *stream, const char *value) {
  const unsigned char *cursor = (const unsigned char *)(value == NULL ? "" : value);
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

static const char *infer_status_name(npunlock_infer_status status) {
  switch (status) {
  case NPUNLOCK_INFER_OK:
    return "ok";
  case NPUNLOCK_INFER_BAD_REQUEST:
    return "bad-request";
  case NPUNLOCK_INFER_OUT_OF_MEMORY:
    return "out-of-memory";
  case NPUNLOCK_INFER_LOADER_NOT_FOUND:
    return "loader-not-found";
  case NPUNLOCK_INFER_SYMBOL_MISSING:
    return "symbol-missing";
  case NPUNLOCK_INFER_NPU_NOT_FOUND:
    return "npu-not-found";
  case NPUNLOCK_INFER_GRAPH_EXTENSION_MISSING:
    return "graph-extension-missing";
  case NPUNLOCK_INFER_UNSUPPORTED:
    return "unsupported";
  case NPUNLOCK_INFER_DRIVER_FAILED:
    return "driver-failed";
  case NPUNLOCK_INFER_BAD_RESULT:
    return "bad-result";
  default:
    return "unknown";
  }
}

static const char *stage_state(const npunlock_infer_result *result, npunlock_infer_stage stage) {
  if (result->stage > stage || result->status == NPUNLOCK_INFER_OK) {
    return "passed";
  }
  if (result->stage == stage) {
    return "failed";
  }
  return "not-run";
}

static int write_report(FILE *stream, const char *bundle_root, const npunlock_infer_result *result,
                        const char *oracle_state) {
  char driver_index[16];
  char device_index[16];
  if (result->driver_index == NPUNLOCK_INFER_AUTO_INDEX) {
    memcpy(driver_index, "null", sizeof("null"));
  } else {
    snprintf(driver_index, sizeof(driver_index), "%" PRIu32, result->driver_index);
  }
  if (result->device_index == NPUNLOCK_INFER_AUTO_INDEX) {
    memcpy(device_index, "null", sizeof("null"));
  } else {
    snprintf(device_index, sizeof(device_index), "%" PRIu32, result->device_index);
  }
  if (fprintf(stream, "{\n  \"schema\": \"npunlock.porting.linux-execution.v1\",\n"
                      "  \"mode\": \"in-process-bring-up\",\n  \"bundle_root\": ") < 0 ||
      !write_json_string(stream, bundle_root) ||
      fprintf(stream,
              ",\n  \"stages\": {\n"
              "    \"loader\": \"%s\",\n"
              "    \"device_selection\": \"%s\",\n"
              "    \"graph_extension\": \"%s\",\n"
              "    \"graph_create\": \"%s\",\n"
              "    \"initialization\": \"%s\",\n"
              "    \"execution\": \"%s\",\n"
              "    \"oracle\": \"%s\"\n"
              "  },\n"
              "  \"result\": {\"status\": \"%s\", \"last_stage\": \"%s\", "
              "\"level_zero_result\": \"0x%08" PRIx32 "\", "
              "\"driver_index\": %s, \"device_index\": %s, "
              "\"driver_version\": %" PRIu32 ", \"vendor_id\": %" PRIu32 ", "
              "\"device_id\": %" PRIu32 "},\n  \"diagnostic\": ",
              stage_state(result, NPUNLOCK_INFER_STAGE_LOADER),
              stage_state(result, NPUNLOCK_INFER_STAGE_DEVICE),
              stage_state(result, NPUNLOCK_INFER_STAGE_GRAPH_EXTENSION),
              stage_state(result, NPUNLOCK_INFER_STAGE_GRAPH_CREATE),
              stage_state(result, NPUNLOCK_INFER_STAGE_INITIALIZATION),
              stage_state(result, NPUNLOCK_INFER_STAGE_EXECUTION), oracle_state,
              infer_status_name(result->status), npunlock_infer_stage_name(result->stage),
              result->driver_result, driver_index, device_index, result->driver_version,
              result->vendor_id, result->device_id) < 0 ||
      !write_json_string(stream, result->diagnostic) ||
      fprintf(stream,
              ",\n  \"scope\": \"Linux in-process compatibility probe; a passed graph-create "
              "stage alone does not establish execution or semantic compatibility\"\n}\n") < 0) {
    return 0;
  }
  return 1;
}

int main(int argc, char **argv) {
  const char *bundle_root = NULL;
  const char *report_path = NULL;
  char *manifest_path = NULL;
  char *graph_path = NULL;
  char *input_path = NULL;
  char *expected_path = NULL;
  file_buffer graph = {0};
  file_buffer input = {0};
  file_buffer expected = {0};
  npunlock_infer_request request = {0};
  npunlock_infer_result result = {0};
  const char *oracle_state = "not-run";
  FILE *report = stdout;
  int index;
  int success = 0;

  if (argc == 2 && strcmp(argv[1], "--version") == 0) {
    printf("npunlock-linux-probe %s\n", NPUNLOCK_VERSION_STRING);
    return 0;
  }
  if (argc == 2 && (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0)) {
    print_usage(argv[0]);
    return 0;
  }
  for (index = 1; index < argc; ++index) {
    if (strcmp(argv[index], "--bundle") == 0 && index + 1 < argc && bundle_root == NULL) {
      bundle_root = argv[++index];
    } else if (strcmp(argv[index], "--report") == 0 && index + 1 < argc && report_path == NULL) {
      report_path = argv[++index];
    } else {
      fprintf(stderr, "unknown or incomplete option: %s\n", argv[index]);
      print_usage(argv[0]);
      return 2;
    }
  }
  if (bundle_root == NULL) {
    fprintf(stderr, "--bundle is required\n");
    return 2;
  }

  manifest_path = join_path(bundle_root, "manifest.json");
  graph_path = join_path(bundle_root, "graph.blob");
  input_path = join_path(bundle_root, "input-0.bin");
  expected_path = join_path(bundle_root, "expected-output-0.bin");
  if (manifest_path == NULL || graph_path == NULL || input_path == NULL || expected_path == NULL ||
      !file_exists(manifest_path)) {
    fprintf(stderr, "bundle paths could not be resolved or manifest.json is missing\n");
    goto cleanup;
  }
  graph = read_file(graph_path, NPUNLOCK_INFER_MAX_GRAPH_SIZE);
  input = read_file(input_path, NPUNLOCK_INFER_MAX_INPUT_SIZE);
  expected = read_file(expected_path, NPUNLOCK_INFER_MAX_OUTPUT_SIZE);
  if (graph.size == SIZE_MAX || input.size == SIZE_MAX || expected.size == SIZE_MAX) {
    goto cleanup;
  }
  if (input.size != 32 || expected.size != 32) {
    fprintf(stderr, "the v1 probe requires exactly 32-byte input and expected-output tensors\n");
    goto cleanup;
  }

  request.driver_index = NPUNLOCK_INFER_AUTO_INDEX;
  request.device_index = NPUNLOCK_INFER_AUTO_INDEX;
  request.graph = graph.data;
  request.graph_size = graph.size;
  request.input_count = 1;
  request.inputs[0].argument_index = 0;
  request.inputs[0].data = input.data;
  request.inputs[0].data_size = input.size;
  npunlock_infer_execute(&request, &result);
  if (result.status == NPUNLOCK_INFER_OK) {
    if (result.output_count == 1 && result.outputs[0].argument_index == 1 &&
        result.outputs[0].data_size == expected.size &&
        memcmp(result.outputs[0].data, expected.data, expected.size) == 0) {
      oracle_state = "passed";
      success = 1;
    } else {
      oracle_state = "failed";
    }
  }

  if (report_path != NULL) {
    report = fopen(report_path, "wb");
    if (report == NULL) {
      fprintf(stderr, "could not open report output: %s\n", report_path);
      success = 0;
      goto cleanup;
    }
  }
  if (!write_report(report, bundle_root, &result, oracle_state) || fflush(report) != 0) {
    fprintf(stderr, "could not write execution report\n");
    success = 0;
    goto cleanup;
  }
  if (report_path != NULL) {
    if (fclose(report) != 0) {
      report = NULL;
      fprintf(stderr, "could not finish execution report: %s\n", report_path);
      success = 0;
      goto cleanup;
    }
    report = NULL;
    printf("wrote execution report to %s\n", report_path);
  }

cleanup:
  if (report_path != NULL && report != NULL && report != stdout) {
    fclose(report);
  }
  npunlock_infer_result_release(&result);
  free(expected.data);
  free(input.data);
  free(graph.data);
  free(expected_path);
  free(input_path);
  free(graph_path);
  free(manifest_path);
  return success ? 0 : 1;
}
