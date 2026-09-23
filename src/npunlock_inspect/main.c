#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "npunlock/patchblob.h"
#include "npunlock/version.h"

#include "internal.h"

typedef struct file_buffer {
  uint8_t *data;
  size_t size;
} file_buffer;

static void print_usage(const char *program) {
  printf("usage: %s --graph GRAPH.blob [--report REPORT.json]\n", program);
  printf("       %s --version\n", program);
}

static file_buffer read_file(const char *path) {
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
    fprintf(stderr, "could not open graph blob: %s\n", path);
    return buffer;
  }
  length = ftell(stream);
  if (length <= 0 || fseek(stream, 0, SEEK_SET) != 0) {
    fclose(stream);
    fprintf(stderr, "graph blob is empty or unreadable: %s\n", path);
    return buffer;
  }
  buffer.data = (uint8_t *)malloc((size_t)length);
  if (buffer.data == NULL || fread(buffer.data, 1, (size_t)length, stream) != (size_t)length) {
    free(buffer.data);
    buffer.data = NULL;
    fclose(stream);
    fprintf(stderr, "could not read graph blob: %s\n", path);
    return buffer;
  }
  buffer.size = (size_t)length;
  fclose(stream);
  return buffer;
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

static const char *precision_name(uint32_t flags) {
  if ((flags & PATCHBLOB_CONTRACT_FP16) != 0 && (flags & PATCHBLOB_CONTRACT_FP32) == 0) {
    return "fp16";
  }
  if ((flags & PATCHBLOB_CONTRACT_FP32) != 0 && (flags & PATCHBLOB_CONTRACT_FP16) == 0) {
    return "fp32";
  }
  return "unknown";
}

static int write_report(FILE *stream, const char *graph_path, const file_buffer *graph,
                        const patchblob_discovery_result *discovery) {
  char graph_hash[65];
  size_t index;
  hash_view((npunlock_view){graph->data, graph->size}, graph_hash);
  if (fprintf(stream,
              "{\n  \"schema\": \"npunlock.porting.inspect.v1\",\n  \"graph\": {\"path\": ") < 0 ||
      !write_json_string(stream, graph_path) ||
      fprintf(stream, ", \"size\": %zu, \"sha256\": \"%s\"},\n", graph->size, graph_hash) < 0 ||
      fprintf(stream,
              "  \"act\": {\"group_count\": %zu, \"target_count\": %zu, "
              "\"targets\": [\n",
              discovery->group_count, discovery->target_count) < 0) {
    return 0;
  }
  for (index = 0; index < discovery->target_count; ++index) {
    const patchblob_discovered_target *discovered = &discovery->targets[index];
    const patchblob_target *target = &discovered->target;
    if (fprintf(stream,
                "    %s{\"group_index\": %u, \"invocation_index\": %u, "
                "\"range_index\": %u, \"input_count\": %u, \"element_count\": "
                "%" PRIu64 ", \"span_bytes\": %" PRIu64 ", \"contract_flags\": %u, "
                "\"precision\": \"%s\"}",
                index == 0 ? "" : ",\n", discovered->group_index, target->invocation_index,
                target->range_index, target->expected_input_count, target->expected_element_count,
                target->expected_span_bytes, target->required_contract_flags,
                precision_name(target->required_contract_flags)) < 0) {
      return 0;
    }
  }
  return fprintf(stream, "\n  ]},\n  \"scope\": \"observed NPU3720/compiler-8.3 ACT contract; "
                         "discovery is not proof of execution compatibility\"\n}\n") >= 0;
}

static void print_diagnostic(const npunlock_diagnostic *diagnostic) {
  if (diagnostic != NULL && diagnostic->json.data != NULL) {
    fwrite(diagnostic->json.data, 1, diagnostic->json.size, stderr);
    if (diagnostic->json.size == 0 || diagnostic->json.data[diagnostic->json.size - 1] != '\n') {
      fputc('\n', stderr);
    }
  }
}

int main(int argc, char **argv) {
  const char *graph_path = NULL;
  const char *report_path = NULL;
  file_buffer graph = {0};
  patchblob_discovery_result discovery = {0};
  npunlock_status status;
  FILE *report = stdout;
  int index;
  int success = 0;

  if (argc == 2 && strcmp(argv[1], "--version") == 0) {
    printf("npunlock-inspect %s\n", NPUNLOCK_VERSION_STRING);
    return 0;
  }
  if (argc == 2 && (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0)) {
    print_usage(argv[0]);
    return 0;
  }
  for (index = 1; index < argc; ++index) {
    if (strcmp(argv[index], "--graph") == 0 && index + 1 < argc && graph_path == NULL) {
      graph_path = argv[++index];
    } else if (strcmp(argv[index], "--report") == 0 && index + 1 < argc && report_path == NULL) {
      report_path = argv[++index];
    } else {
      fprintf(stderr, "unknown or incomplete option: %s\n", argv[index]);
      print_usage(argv[0]);
      return 2;
    }
  }
  if (graph_path == NULL || (report_path != NULL && strcmp(graph_path, report_path) == 0)) {
    fprintf(stderr, "--graph is required and must not also be the report path\n");
    print_usage(argv[0]);
    return 2;
  }

  graph = read_file(graph_path);
  if (graph.size == SIZE_MAX) {
    goto cleanup;
  }
  status = patchblob_discover_targets((npunlock_view){graph.data, graph.size}, &discovery);
  if (status != NPUNLOCK_STATUS_OK) {
    print_diagnostic(&discovery.diagnostic);
    goto cleanup;
  }
  if (report_path != NULL) {
#ifdef _WIN32
    if (fopen_s(&report, report_path, "wb") != 0) {
      report = NULL;
    }
#else
    report = fopen(report_path, "wb");
#endif
    if (report == NULL) {
      fprintf(stderr, "could not open report output: %s\n", report_path);
      goto cleanup;
    }
  }
  if (!write_report(report, graph_path, &graph, &discovery) || fflush(report) != 0) {
    fprintf(stderr, "could not write inspection report\n");
    goto cleanup;
  }
  if (report_path != NULL) {
    if (fclose(report) != 0) {
      report = NULL;
      fprintf(stderr, "could not finish inspection report: %s\n", report_path);
      goto cleanup;
    }
    report = NULL;
    printf("wrote inspection report to %s\n", report_path);
  }
  success = 1;

cleanup:
  if (report_path != NULL && report != NULL) {
    fclose(report);
  }
  patchblob_discovery_result_release(&discovery);
  free(graph.data);
  return success ? 0 : 1;
}
