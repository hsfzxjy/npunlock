#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "npunlock/patchblob.h"
#include "npunlock/version.h"

#include "graph_blob.h"
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

static const char *tensor_precision_name(uint32_t precision) {
  if (precision == PATCHBLOB_TENSOR_PRECISION_FP16_V2) {
    return "fp16";
  }
  if (precision == PATCHBLOB_TENSOR_PRECISION_FP32_V2) {
    return "fp32";
  }
  if (precision == PATCHBLOB_TENSOR_PRECISION_I32_V2) {
    return "i32";
  }
  if (precision == PATCHBLOB_TENSOR_PRECISION_I16_V2) {
    return "i16";
  }
  if (precision == PATCHBLOB_TENSOR_PRECISION_I8_V2) {
    return "i8";
  }
  if (precision == PATCHBLOB_TENSOR_PRECISION_U8_V2) {
    return "u8";
  }
  if (precision == PATCHBLOB_TENSOR_PRECISION_U16_V2) {
    return "u16";
  }
  if (precision == PATCHBLOB_TENSOR_PRECISION_U32_V2) {
    return "u32";
  }
  return "unknown";
}

static const char *target_precision_name(const patchblob_target_v2 *target) {
  uint32_t precision = target->tensors[0].precision;
  size_t index;
  for (index = 1; index < target->tensor_count; ++index) {
    if (target->tensors[index].precision != precision) {
      return "mixed";
    }
  }
  return tensor_precision_name(precision);
}

static int write_report(FILE *stream, const char *graph_path, const file_buffer *graph,
                        const npunlock_graph_routing_observations *routing) {
  char graph_hash[65];
  size_t index;
  hash_view((npunlock_view){graph->data, graph->size}, graph_hash);
  if (fprintf(stream,
              "{\n  \"schema\": \"npunlock.porting.inspect.v7\",\n  \"graph\": {\"path\": ") < 0 ||
      !write_json_string(stream, graph_path) ||
      fprintf(stream, ", \"size\": %zu, \"sha256\": \"%s\"},\n", graph->size, graph_hash) < 0 ||
      fprintf(stream, "  \"graph_contract\": {\"compiler\": \"8.3\", \"npu\": \"3720\", "
                      "\"loader_elf\": \"1.2.0\", \"mapped_inference\": \"7.0.4\", "
                      "\"source\": \"compatibility_string + version notes\"},\n") < 0 ||
      fprintf(stream,
              "  \"act\": {\"group_count\": %zu, \"target_count\": %zu, "
              "\"tile_count_observed\": %u, "
              "\"targets\": [\n",
              routing->group_count, routing->invocation_count, routing->tile_count_observed) < 0) {
    return 0;
  }
  for (index = 0; index < routing->invocation_count; ++index) {
    const npunlock_invocation_routing_observation *observation = &routing->invocations[index];
    const patchblob_target_v2 *target = &observation->target;
    const patchblob_tensor_contract_v2 *output = &target->tensors[target->tensor_count - 1u];
    size_t tensor_index;
    bool first_scalar = true;
    bool input_1_scalar = target->tensor_count == 3u && output->element_count > 1u &&
                          target->tensors[1].element_count == 1u;
    if (fprintf(stream,
                "%s    {\"group_index\": %u, \"invocation_index\": %u, "
                "\"range_index\": %u, \"input_count\": %u, \"element_count\": "
                "%" PRIu64 ", \"span_bytes\": %" PRIu64 ", \"contract_version\": 2, "
                "\"target_flags\": %u, \"precision\": \"%s\", "
                "\"input_1_scalar\": %s, \"routing_observation\": "
                "{\"tile_candidate_raw\": %u, \"parameter_base\": %" PRIu64
                ", \"wait_mask_raw\": %" PRIu64 ", \"post_mask_raw\": %" PRIu64
                ", \"barrier_group_raw\": %u, \"barrier_mask_raw\": %u, "
                "\"start_after_raw\": %u, \"clean_after_raw\": %u}, "
                "\"scalar_input_indices\": [",
                index == 0 ? "" : ",\n", observation->group_index, target->invocation_index,
                target->range_index, target->tensor_count - 1u, output->element_count,
                output->span_bytes, target->target_flags, target_precision_name(target),
                input_1_scalar ? "true" : "false", observation->tile_candidate_raw,
                observation->parameter_base, observation->wait_mask_raw, observation->post_mask_raw,
                observation->barrier_group_raw, observation->barrier_mask_raw,
                observation->start_after_raw, observation->clean_after_raw) < 0) {
      return 0;
    }
    for (tensor_index = 0; tensor_index + 1u < target->tensor_count; ++tensor_index) {
      const patchblob_tensor_contract_v2 *tensor = &target->tensors[tensor_index];
      if (output->element_count > 1u && tensor->element_count == 1u) {
        if (fprintf(stream, "%s%u", first_scalar ? "" : ", ", tensor->tensor_index) < 0) {
          return 0;
        }
        first_scalar = false;
      }
    }
    if (fprintf(stream, "], \"tensors\": [") < 0) {
      return 0;
    }
    for (tensor_index = 0; tensor_index < target->tensor_count; ++tensor_index) {
      const patchblob_tensor_contract_v2 *tensor = &target->tensors[tensor_index];
      const npunlock_tensor_routing_observation *tensor_routing =
          &observation->tensors[tensor_index];
      const char *role = tensor->role == PATCHBLOB_TENSOR_ROLE_INPUT_V2 ? "input" : "output";
      size_t dimension_index;
      if (fprintf(stream,
                  "%s{\"role\": \"%s\", \"index\": %u, \"precision\": \"%s\", "
                  "\"element_count\": %" PRIu64 ", \"span_bytes\": %" PRIu64
                  ", \"observed_flags\": %u, \"routing_observation\": "
                  "{\"data_symbol_raw\": %u, \"data_addend_raw\": %" PRId64
                  ", \"rank\": %u, \"dimensions\": [",
                  tensor_index == 0 ? "" : ", ", role, tensor->tensor_index,
                  tensor_precision_name(tensor->precision), tensor->element_count,
                  tensor->span_bytes, tensor->observed_flags, tensor_routing->data_symbol,
                  tensor_routing->data_addend, tensor_routing->rank) < 0) {
        return 0;
      }
      for (dimension_index = 0; dimension_index < tensor_routing->rank; ++dimension_index) {
        if (fprintf(stream, "%s%d", dimension_index == 0 ? "" : ", ",
                    tensor_routing->dimensions[dimension_index]) < 0) {
          return 0;
        }
      }
      if (fprintf(stream, "], \"strides_bits\": [") < 0) {
        return 0;
      }
      for (dimension_index = 0; dimension_index < tensor_routing->rank; ++dimension_index) {
        if (fprintf(stream, "%s%" PRId64, dimension_index == 0 ? "" : ", ",
                    tensor_routing->strides_bits[dimension_index]) < 0) {
          return 0;
        }
      }
      if (fprintf(stream, "]}}") < 0) {
        return 0;
      }
    }
    if (fprintf(stream, "]}") < 0) {
      return 0;
    }
  }
  if (fprintf(stream, "\n  ], \"act_edge_candidates\": [\n") < 0) {
    return 0;
  }
  for (index = 0; index < routing->act_edge_candidate_count; ++index) {
    const npunlock_act_edge_candidate_observation *candidate = &routing->act_edge_candidates[index];
    if (fprintf(stream,
                "%s    {\"producer_group_index\": %u, \"producer_invocation_index\": %u, "
                "\"producer_tensor_record_index\": %u, \"consumer_group_index\": %u, "
                "\"consumer_invocation_index\": %u, \"consumer_tensor_record_index\": %u, "
                "\"consumer_candidate_count\": %u, \"match_observation\": "
                "{\"data_symbol_raw\": %u, \"data_addend_raw\": %" PRId64
                ", \"span_bytes\": %" PRIu64 ", \"schedule_hop_count\": %u, "
                "\"barrier_intersection_raw\": %" PRIu64 "}}",
                index == 0 ? "" : ",\n", candidate->producer_group_index,
                candidate->producer_invocation_index, candidate->producer_tensor_record_index,
                candidate->consumer_group_index, candidate->consumer_invocation_index,
                candidate->consumer_tensor_record_index, candidate->consumer_candidate_count,
                candidate->data_symbol_raw, candidate->data_addend_raw, candidate->span_bytes,
                candidate->schedule_hop_count, candidate->barrier_intersection_raw) < 0) {
      return 0;
    }
  }
  if (fprintf(stream, "\n  ], \"network_output_routes\": [\n") < 0) {
    return 0;
  }
  for (index = 0; index < routing->network_output_route_count; ++index) {
    const npunlock_network_output_route_observation *route = &routing->network_output_routes[index];
    if (fprintf(stream,
                "%s    {\"dma_section_index\": %u, \"dma_task_index\": %u, "
                "\"cmx_source_symbol_raw\": %u, \"cmx_source_addend_raw\": %" PRId64
                ", \"output_symbol_raw\": %u, \"output_addend_raw\": %" PRId64
                ", \"length_bytes\": %u}",
                index == 0 ? "" : ",\n", route->dma_section_index, route->dma_task_index,
                route->cmx_source_symbol_raw, route->cmx_source_addend_raw,
                route->output_symbol_raw, route->output_addend_raw, route->length_bytes) < 0) {
      return 0;
    }
  }
  return fprintf(stream,
                 "\n  ]},\n  \"scope\": \"observed NPU3720/compiler-8.3 ACT and network-output "
                 "DMA contract; routing and scheduling fields are raw validated observations; "
                 "ACT edge candidates are exact interval/ACT-schedule correlations, including "
                 "bounded transitive wait/post paths, not semantic graph edges, logical offsets, "
                 "partitions, or replicas; discovery is not proof of "
                 "execution compatibility\"\n}\n") >= 0;
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
  npunlock_graph_routing_observations routing = {0};
  npunlock_diagnostic diagnostic = {0};
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
  status = npunlock_inspect_graph_routing((npunlock_view){graph.data, graph.size}, &routing,
                                          &diagnostic);
  if (status != NPUNLOCK_STATUS_OK) {
    print_diagnostic(&diagnostic);
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
  if (!write_report(report, graph_path, &graph, &routing) || fflush(report) != 0) {
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
  npunlock_graph_routing_observations_release(&routing);
  npunlock_diagnostic_release(&diagnostic);
  free(graph.data);
  return success ? 0 : 1;
}
