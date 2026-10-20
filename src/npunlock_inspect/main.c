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
  printf("       %s --graph BASE.blob --compare CANDIDATE.blob [--report REPORT.json]\n", program);
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

static void format_digest(const uint8_t digest[32], char text[65]) {
  static const char digits[] = "0123456789abcdef";
  size_t index;
  for (index = 0; index < 32u; ++index) {
    text[index * 2] = digits[digest[index] >> 4];
    text[index * 2 + 1] = digits[digest[index] & 15u];
  }
  text[64] = 0;
}

static void hash_view(npunlock_view view, char text[65]) {
  uint8_t digest[32];
  npunlock_sha256(view, digest);
  format_digest(digest, text);
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

static const char *dma_endpoint_kind_name(uint32_t kind) {
  switch (kind) {
  case NPUNLOCK_DMA_ENDPOINT_CMX:
    return "cmx";
  case NPUNLOCK_DMA_ENDPOINT_DDR_BUFFER:
    return "ddr_buffer";
  case NPUNLOCK_DMA_ENDPOINT_CONSTANT:
    return "constant";
  case NPUNLOCK_DMA_ENDPOINT_NETWORK_INPUT:
    return "network_input";
  case NPUNLOCK_DMA_ENDPOINT_NETWORK_OUTPUT:
    return "network_output";
  case NPUNLOCK_DMA_ENDPOINT_RUNTIME:
    return "runtime";
  default:
    return "unknown";
  }
}

static int write_report(FILE *stream, const char *graph_path, const file_buffer *graph,
                        const npunlock_graph_routing_observations *routing) {
  char graph_hash[65];
  size_t index;
  hash_view((npunlock_view){graph->data, graph->size}, graph_hash);
  if (fprintf(stream,
              "{\n  \"schema\": \"npunlock.porting.inspect.v19\",\n  \"graph\": {\"path\": ") < 0 ||
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
    char code_hash[65];
    size_t tensor_index;
    bool first_scalar = true;
    bool input_1_scalar = target->tensor_count == 3u && output->element_count > 1u &&
                          target->tensors[1].element_count == 1u;
    format_digest(observation->code_sha256, code_hash);
    if (fprintf(stream,
                "%s    {\"group_index\": %u, \"invocation_index\": %u, "
                "\"range_index\": %u, \"input_count\": %u, \"element_count\": "
                "%" PRIu64 ", \"span_bytes\": %" PRIu64 ", \"contract_version\": 2, "
                "\"target_flags\": %u, \"precision\": \"%s\", "
                "\"input_1_scalar\": %s, \"code_observation\": "
                "{\"offset_raw\": %" PRIu64 ", \"extent\": %u, \"sha256\": \"%s\"}, "
                "\"routing_observation\": "
                "{\"tile_candidate_raw\": %u, \"parameter_base\": %" PRIu64
                ", \"wait_mask_raw\": %" PRIu64 ", \"post_mask_raw\": %" PRIu64
                ", \"barrier_group_raw\": %u, \"barrier_mask_raw\": %u, "
                "\"start_after_raw\": %u, \"clean_after_raw\": %u}, "
                "\"scalar_input_indices\": [",
                index == 0 ? "" : ",\n", observation->group_index, target->invocation_index,
                target->range_index, target->tensor_count - 1u, output->element_count,
                output->span_bytes, target->target_flags, target_precision_name(target),
                input_1_scalar ? "true" : "false", observation->code_offset_raw,
                observation->code_extent, code_hash, observation->tile_candidate_raw,
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
  if (fprintf(stream, "\n  ], \"act_consumer_cover_observations\": [\n") < 0) {
    return 0;
  }
  for (index = 0; index < routing->act_consumer_cover_count; ++index) {
    const npunlock_act_consumer_cover_observation *observation =
        &routing->act_consumer_covers[index];
    if (fprintf(stream,
                "%s    {\"producer_group_index\": %u, \"producer_invocation_count\": %u, "
                "\"matched_producer_invocation_count\": %u, "
                "\"unresolved_producer_invocation_count\": %u, \"consumer_group_count\": %u, "
                "\"consumer_input_count\": %u, \"exact_cover\": %s, \"unresolved_flags\": %u}",
                index == 0 ? "" : ",\n", observation->producer_group_index,
                observation->producer_invocation_count,
                observation->matched_producer_invocation_count,
                observation->unresolved_producer_invocation_count,
                observation->consumer_group_count, observation->consumer_input_count,
                observation->exact_cover ? "true" : "false", observation->unresolved_flags) < 0) {
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
  if (fprintf(stream, "\n  ], \"output_slice_candidates\": [\n") < 0) {
    return 0;
  }
  for (index = 0; index < routing->act_output_slice_candidate_count; ++index) {
    const npunlock_act_output_slice_candidate_observation *candidate =
        &routing->act_output_slice_candidates[index];
    if (fprintf(stream,
                "%s    {\"act_group_index\": %u, \"act_invocation_index\": %u, "
                "\"act_tensor_record_index\": %u, \"dma_section_index\": %u, "
                "\"dma_task_index\": %u, \"route_candidate_count\": %u, "
                "\"match_observation\": {\"cmx_source_symbol_raw\": %u, "
                "\"act_data_addend_raw\": %" PRId64 ", \"act_span_bytes\": %" PRIu64
                ", \"route_cmx_source_addend_raw\": %" PRId64
                ", \"route_length_bytes\": %u, \"output_symbol_raw\": %u, "
                "\"route_output_addend_raw\": %" PRId64
                ", \"output_slice_addend_candidate\": %" PRIu64 "}}",
                index == 0 ? "" : ",\n", candidate->act_group_index,
                candidate->act_invocation_index, candidate->act_tensor_record_index,
                candidate->dma_section_index, candidate->dma_task_index,
                candidate->route_candidate_count, candidate->cmx_source_symbol_raw,
                candidate->act_data_addend_raw, candidate->act_span_bytes,
                candidate->route_cmx_source_addend_raw, candidate->route_length_bytes,
                candidate->output_symbol_raw, candidate->route_output_addend_raw,
                candidate->output_slice_addend_candidate) < 0) {
      return 0;
    }
  }
  if (fprintf(stream, "\n  ], \"output_cover_observations\": [\n") < 0) {
    return 0;
  }
  for (index = 0; index < routing->act_output_cover_count; ++index) {
    const npunlock_act_output_cover_observation *cover = &routing->act_output_covers[index];
    if (fprintf(stream,
                "%s    {\"output_symbol_raw\": %u, \"act_group_count\": %u, "
                "\"group_invocation_count\": %u, \"routed_invocation_count\": %u, "
                "\"unrouted_invocation_count\": %u, \"route_count\": %u, "
                "\"slice_candidate_count\": %u, \"route_span_bytes\": %" PRIu64
                ", \"slice_span_bytes\": %" PRIu64 ", \"exact_cover\": %s, "
                "\"unresolved_flags\": %u}",
                index == 0 ? "" : ",\n", cover->output_symbol_raw, cover->act_group_count,
                cover->group_invocation_count, cover->routed_invocation_count,
                cover->unrouted_invocation_count, cover->route_count, cover->slice_candidate_count,
                cover->route_span_bytes, cover->slice_span_bytes,
                cover->exact_cover ? "true" : "false", cover->unresolved_flags) < 0) {
      return 0;
    }
  }
  if (fprintf(stream, "\n  ], \"replica_group_observations\": [\n") < 0) {
    return 0;
  }
  for (index = 0; index < routing->act_replica_group_count; ++index) {
    const npunlock_act_replica_group_observation *group = &routing->act_replica_groups[index];
    if (fprintf(stream,
                "%s    {\"act_group_index\": %u, \"routed_invocation_count\": %u, "
                "\"unrouted_invocation_count\": %u, \"translation_solution_count\": %u, "
                "\"cmx_translation_candidate\": %" PRId64 "}",
                index == 0 ? "" : ",\n", group->act_group_index, group->routed_invocation_count,
                group->unrouted_invocation_count, group->translation_solution_count,
                group->cmx_translation_candidate) < 0) {
      return 0;
    }
  }
  if (fprintf(stream, "\n  ], \"replica_candidates\": [\n") < 0) {
    return 0;
  }
  for (index = 0; index < routing->act_replica_candidate_count; ++index) {
    const npunlock_act_replica_candidate_observation *candidate =
        &routing->act_replica_candidates[index];
    if (fprintf(stream,
                "%s    {\"act_group_index\": %u, \"unrouted_invocation_index\": %u, "
                "\"routed_invocation_index\": %u, \"unrouted_tile_candidate_raw\": %u, "
                "\"routed_tile_candidate_raw\": %u, \"output_symbol_raw\": %u, "
                "\"output_slice_addend_candidate\": %" PRIu64
                ", \"output_slice_span_bytes\": %" PRIu64
                ", \"cmx_translation_candidate\": %" PRId64 "}",
                index == 0 ? "" : ",\n", candidate->act_group_index,
                candidate->unrouted_invocation_index, candidate->routed_invocation_index,
                candidate->unrouted_tile_candidate_raw, candidate->routed_tile_candidate_raw,
                candidate->output_symbol_raw, candidate->output_slice_addend_candidate,
                candidate->output_slice_span_bytes, candidate->cmx_translation_candidate) < 0) {
      return 0;
    }
  }
  if (fprintf(stream, "\n  ], \"group_completeness_observations\": [\n") < 0) {
    return 0;
  }
  for (index = 0; index < routing->act_group_completeness_count; ++index) {
    const npunlock_act_group_completeness_observation *observation =
        &routing->act_group_completeness[index];
    if (fprintf(stream,
                "%s    {\"act_group_index\": %u, \"output_symbol_raw\": %u, "
                "\"invocation_count\": %u, \"routed_invocation_count\": %u, "
                "\"replica_invocation_count\": %u, \"unresolved_invocation_count\": %u, "
                "\"complete\": %s, \"unresolved_flags\": %u}",
                index == 0 ? "" : ",\n", observation->act_group_index,
                observation->output_symbol_raw, observation->invocation_count,
                observation->routed_invocation_count, observation->replica_invocation_count,
                observation->unresolved_invocation_count, observation->complete ? "true" : "false",
                observation->unresolved_flags) < 0) {
      return 0;
    }
  }
  if (fprintf(stream, "\n  ]},\n  \"dma\": {\"task_count\": %zu, \"tasks\": [\n",
              routing->dma_task_count) < 0) {
    return 0;
  }
  for (index = 0; index < routing->dma_task_count; ++index) {
    const npunlock_dma_task_observation *task = &routing->dma_tasks[index];
    if (fprintf(stream,
                "%s    {\"section_index\": %u, \"task_index\": %u, \"engine_index\": %u, "
                "\"transfer_observation\": {\"type_raw\": %u, \"length_bytes\": %u, "
                "\"num_planes_raw\": %u, \"task_id_raw\": %u, "
                "\"source_plane_stride_raw\": %d, \"destination_plane_stride_raw\": %d, "
                "\"source_width_raw\": %u, \"source_stride_raw\": %d, "
                "\"destination_width_raw\": %u, \"destination_stride_raw\": %d}, "
                "\"scheduling_observation\": {\"wait_mask_raw\": %" PRIu64
                ", \"post_mask_raw\": %" PRIu64
                ", \"start_after_raw\": %u, \"clean_after_raw\": %u}, "
                "\"source\": {\"memory\": \"%s\", \"value_raw\": %" PRIu64
                ", \"relocation_section_index\": %u, \"symbol_table_index\": %u, "
                "\"symbol_raw\": %u, \"relocation_type\": %u, \"addend_raw\": %" PRId64
                "}, \"destination\": {\"memory\": \"%s\", \"value_raw\": %" PRIu64
                ", \"relocation_section_index\": %u, \"symbol_table_index\": %u, "
                "\"symbol_raw\": %u, \"relocation_type\": %u, \"addend_raw\": %" PRId64 "}}",
                index == 0 ? "" : ",\n", task->section_index, task->task_index, task->engine_index,
                task->transfer_type_raw, task->length_bytes, task->num_planes_raw,
                task->task_id_raw, task->source_plane_stride_raw,
                task->destination_plane_stride_raw, task->source_width_raw, task->source_stride_raw,
                task->destination_width_raw, task->destination_stride_raw, task->wait_mask_raw,
                task->post_mask_raw, task->start_after_raw, task->clean_after_raw,
                dma_endpoint_kind_name(task->source.memory_kind), task->source_value_raw,
                task->source.relocation_section_index, task->source.symbol_table_index,
                task->source.symbol_raw, task->source.relocation_type, task->source.addend_raw,
                dma_endpoint_kind_name(task->destination.memory_kind), task->destination_value_raw,
                task->destination.relocation_section_index, task->destination.symbol_table_index,
                task->destination.symbol_raw, task->destination.relocation_type,
                task->destination.addend_raw) < 0) {
      return 0;
    }
  }
  if (fprintf(stream, "\n  ], \"act_source_candidates\": [\n") < 0) {
    return 0;
  }
  for (index = 0; index < routing->act_dma_source_candidate_count; ++index) {
    const npunlock_act_dma_source_candidate_observation *candidate =
        &routing->act_dma_source_candidates[index];
    if (fprintf(stream,
                "%s    {\"act_group_index\": %u, \"act_invocation_index\": %u, "
                "\"act_tensor_record_index\": %u, \"dma_section_index\": %u, "
                "\"dma_task_index\": %u, \"dma_engine_index\": %u, "
                "\"dma_candidate_count\": %u, \"match_observation\": {"
                "\"ddr_symbol_table_index\": %u, \"ddr_symbol_raw\": %u, "
                "\"act_data_addend_raw\": %" PRId64 ", \"dma_source_addend_raw\": %" PRId64
                ", \"ddr_destination_addend_raw\": %" PRId64 ", \"span_bytes\": %" PRIu64
                ", \"barrier_intersection_raw\": %" PRIu64 ", \"dma_post_mask_raw\": %" PRIu64 "}}",
                index == 0 ? "" : ",\n", candidate->act_group_index,
                candidate->act_invocation_index, candidate->act_tensor_record_index,
                candidate->dma_section_index, candidate->dma_task_index,
                candidate->dma_engine_index, candidate->dma_candidate_count,
                candidate->ddr_symbol_table_index, candidate->ddr_symbol_raw,
                candidate->act_data_addend_raw, candidate->dma_source_addend_raw,
                candidate->ddr_destination_addend_raw, candidate->span_bytes,
                candidate->barrier_intersection_raw, candidate->dma_post_mask_raw) < 0) {
      return 0;
    }
  }
  if (fprintf(stream, "\n  ], \"act_fanin_observations\": [\n") < 0) {
    return 0;
  }
  for (index = 0; index < routing->act_dma_fanin_observation_count; ++index) {
    const npunlock_act_dma_fanin_observation *observation =
        &routing->act_dma_fanin_observations[index];
    if (fprintf(
            stream,
            "%s    {\"act_group_index\": %u, \"invocation_count\": %u, "
            "\"matched_invocation_count\": %u, \"source_dma_task_count\": %u, "
            "\"fanin_candidate_count\": %u, \"fanin_dma_section_index\": %u, "
            "\"fanin_dma_task_index\": %u, \"fanin_dma_engine_index\": %u, "
            "\"ddr_symbol_table_index\": %u, \"ddr_symbol_raw\": %u, "
            "\"ddr_source_addend_raw\": %" PRId64 ", \"span_bytes\": %" PRIu64
            ", \"cmx_destination_addend_raw\": %" PRId64 ", \"cmx_destination_value_raw\": %" PRIu64
            ", \"fanin_wait_mask_raw\": %" PRIu64 ", \"fanin_post_mask_raw\": %" PRIu64
            ", \"exact_cover\": %s, \"unresolved_flags\": %u}",
            index == 0 ? "" : ",\n", observation->act_group_index, observation->invocation_count,
            observation->matched_invocation_count, observation->source_dma_task_count,
            observation->fanin_candidate_count, observation->fanin_dma_section_index,
            observation->fanin_dma_task_index, observation->fanin_dma_engine_index,
            observation->ddr_symbol_table_index, observation->ddr_symbol_raw,
            observation->ddr_source_addend_raw, observation->span_bytes,
            observation->cmx_destination_addend_raw, observation->cmx_destination_value_raw,
            observation->fanin_wait_mask_raw, observation->fanin_post_mask_raw,
            observation->exact_cover ? "true" : "false", observation->unresolved_flags) < 0) {
      return 0;
    }
  }
  if (fprintf(stream, "\n  ]},\n  \"dpu\": {\"invariant_count\": %zu, \"invariants\": [\n",
              routing->dpu_invariant_count) < 0) {
    return 0;
  }
  for (index = 0; index < routing->dpu_invariant_count; ++index) {
    const npunlock_dpu_invariant_observation *invariant = &routing->dpu_invariants[index];
    size_t address_index;
    if (fprintf(stream,
                "%s    {\"section_index\": %u, \"invariant_index\": %u, "
                "\"routing_observation\": {\"cluster_raw\": %u, \"wait_mask_raw\": %" PRIu64
                ", \"post_mask_raw\": %" PRIu64
                ", \"barrier_group_raw\": %u, \"barrier_mask_raw\": %u, "
                "\"start_after_raw\": %u, \"clean_after_raw\": %u, "
                "\"variant_count_raw\": %u}, "
                "\"input_tensor_observation\": {\"dimensions_xyz\": [%u, %u, %u], "
                "\"dtype_raw\": %u, \"dense_raw\": %u, \"element_count\": %" PRIu64
                ", \"span_bytes\": %" PRIu64 "}, "
                "\"output_tensor_observation\": {\"dimensions_xyz\": [%u, %u, %u], "
                "\"dtype_raw\": %u, \"sparse_raw\": %u, \"element_count\": %" PRIu64
                ", \"span_bytes\": %" PRIu64 "}, \"input_cmx_relocations\": [",
                index == 0 ? "" : ",\n", invariant->section_index, invariant->invariant_index,
                invariant->cluster_raw, invariant->wait_mask_raw, invariant->post_mask_raw,
                invariant->barrier_group_raw, invariant->barrier_mask_raw,
                invariant->start_after_raw, invariant->clean_after_raw,
                invariant->variant_count_raw, invariant->input_dimensions_xyz[0],
                invariant->input_dimensions_xyz[1], invariant->input_dimensions_xyz[2],
                invariant->input_dtype_raw, invariant->input_dense_raw,
                invariant->input_element_count, invariant->input_span_bytes,
                invariant->output_dimensions_xyz[0], invariant->output_dimensions_xyz[1],
                invariant->output_dimensions_xyz[2], invariant->output_dtype_raw,
                invariant->output_sparse_raw, invariant->output_element_count,
                invariant->output_span_bytes) < 0) {
      return 0;
    }
    for (address_index = 0; address_index < invariant->input_count; ++address_index) {
      const npunlock_dpu_cmx_address_observation *address = &invariant->inputs[address_index];
      if (fprintf(stream,
                  "%s{\"field_index\": %u, \"register_offset\": %u, "
                  "\"relocation_type\": %u, \"data_symbol_raw\": %u, "
                  "\"data_addend_raw\": %" PRId64 "}",
                  address_index == 0 ? "" : ", ", address->field_index,
                  0x5cu + address->field_index * 4u, address->relocation_type,
                  address->data_symbol_raw, address->data_addend_raw) < 0) {
        return 0;
      }
    }
    if (fprintf(stream, "], \"output_cmx_relocations\": [") < 0) {
      return 0;
    }
    for (address_index = 0; address_index < invariant->output_count; ++address_index) {
      const npunlock_dpu_cmx_address_observation *address = &invariant->outputs[address_index];
      if (fprintf(stream,
                  "%s{\"field_index\": %u, \"register_offset\": %u, "
                  "\"relocation_type\": %u, \"data_symbol_raw\": %u, "
                  "\"data_addend_raw\": %" PRId64 "}",
                  address_index == 0 ? "" : ", ", address->field_index,
                  0xa0u + address->field_index * 4u, address->relocation_type,
                  address->data_symbol_raw, address->data_addend_raw) < 0) {
        return 0;
      }
    }
    if (fprintf(stream, "]}") < 0) {
      return 0;
    }
  }
  if (fprintf(stream, "\n  ], \"act_dpu_edge_candidates\": [\n") < 0) {
    return 0;
  }
  for (index = 0; index < routing->act_dpu_edge_candidate_count; ++index) {
    const npunlock_act_dpu_edge_candidate_observation *candidate =
        &routing->act_dpu_edge_candidates[index];
    const char *direction =
        candidate->direction == NPUNLOCK_ACT_DPU_DIRECTION_ACT_TO_DPU ? "act_to_dpu" : "dpu_to_act";
    if (fprintf(stream,
                "%s    {\"direction\": \"%s\", \"act_group_index\": %u, "
                "\"act_invocation_index\": %u, \"act_tensor_record_index\": %u, "
                "\"dpu_invariant_index\": %u, \"dpu_field_mask_raw\": %u, "
                "\"act_candidate_count\": %u, \"dpu_section_index\": %u, "
                "\"match_observation\": "
                "{\"data_symbol_raw\": %u, \"act_data_addend_raw\": %" PRId64
                ", \"act_span_bytes\": %" PRIu64 ", \"dpu_data_addend_raw\": %" PRId64
                ", \"dpu_span_bytes\": %" PRIu64 ", \"barrier_intersection_raw\": %" PRIu64 "}}",
                index == 0 ? "" : ",\n", direction, candidate->act_group_index,
                candidate->act_invocation_index, candidate->act_tensor_record_index,
                candidate->dpu_invariant_index, candidate->dpu_field_mask_raw,
                candidate->act_candidate_count, candidate->dpu_section_index,
                candidate->data_symbol_raw, candidate->act_data_addend_raw,
                candidate->act_span_bytes, candidate->dpu_data_addend_raw,
                candidate->dpu_span_bytes, candidate->barrier_intersection_raw) < 0) {
      return 0;
    }
  }
  if (fprintf(stream, "\n  ], \"act_dpu_output_cover_observations\": [\n") < 0) {
    return 0;
  }
  for (index = 0; index < routing->act_dpu_output_cover_count; ++index) {
    const npunlock_act_dpu_output_cover_observation *observation =
        &routing->act_dpu_output_covers[index];
    if (fprintf(stream,
                "%s    {\"act_group_index\": %u, \"invocation_count\": %u, "
                "\"matched_invocation_count\": %u, \"unresolved_invocation_count\": %u, "
                "\"dpu_input_count\": %u, \"exact_cover_input_count\": %u, "
                "\"exact_cover\": %s, \"unresolved_flags\": %u}",
                index == 0 ? "" : ",\n", observation->act_group_index,
                observation->invocation_count, observation->matched_invocation_count,
                observation->unresolved_invocation_count, observation->dpu_input_count,
                observation->exact_cover_input_count, observation->exact_cover ? "true" : "false",
                observation->unresolved_flags) < 0) {
      return 0;
    }
  }
  return fprintf(stream,
                 "\n  ]},\n  \"scope\": \"observed NPU3720/compiler-8.3 ACT, DPU-invariant, "
                 "network-output DMA, and raw DMA-task contract; DMA endpoints "
                 "come from validated relocations, while transfer geometry and scheduling "
                 "fields remain raw observations; ACT/DMA source candidates require an exact "
                 "CMX interval and direct schedule match; fan-in observations additionally "
                 "require unique source tasks, an exact non-overlapping DDR union, and one "
                 "scheduled DDR-to-multicast-CMX task, but do not establish DPU ownership; ACT "
                 "edge candidates are exact "
                 "interval/ACT-schedule "
                 "correlations, including bounded transitive wait/post paths; ACT/DPU candidates "
                 "require interval containment and a direct schedule dependency; ACT "
                 "consumer-cover observations require every producer invocation to have one "
                 "unique whole-tensor ACT consumer input; output-slice "
                 "candidates translate only ACT output subranges contained by validated "
                 "network-output DMA routes; output-cover observations require unique, "
                 "non-overlapping slices with the same interval union as the output routes; "
                 "replica candidates require a unique complete CMX-translation bijection; "
                 "group completeness requires an exact output cover and one routed or replica "
                 "classification for every participating invocation; "
                 "ACT/DPU output-cover observations require every group invocation to have one "
                 "unique DPU consumer and exact non-overlapping coverage of each DPU input; "
                 "code fingerprints cover validated range slices but are not semantic operation "
                 "identifiers; DMA observations do not establish semantic graph edges; "
                 "DPU CMX "
                 "relocations and dense tensor extents identify architectural fields and byte "
                 "intervals; candidates do not classify unrouted invocations as replicas; "
                 "discovery is not proof of execution compatibility\"\n}\n") >= 0;
}

static int write_mismatch_names(FILE *stream, uint32_t flags) {
  static const struct {
    uint32_t flag;
    const char *name;
  } names[] = {
      {NPUNLOCK_COMPARE_STRUCTURE_CHANGED, "structure"},
      {NPUNLOCK_COMPARE_INVOCATION_CONTRACT_CHANGED, "invocation_contract"},
      {NPUNLOCK_COMPARE_OUTPUT_ROUTING_CHANGED, "output_routing"},
      {NPUNLOCK_COMPARE_ACT_RELATIONS_CHANGED, "act_relations"},
      {NPUNLOCK_COMPARE_DPU_CHANGED, "dpu"},
      {NPUNLOCK_COMPARE_ACT_DPU_RELATIONS_CHANGED, "act_dpu_relations"},
      {NPUNLOCK_COMPARE_PARTIAL_GROUP_CODE_CHANGE, "partial_group_code_change"},
      {NPUNLOCK_COMPARE_DMA_CHANGED, "dma"},
      {NPUNLOCK_COMPARE_ACT_DMA_RELATIONS_CHANGED, "act_dma_relations"},
  };
  bool first = true;
  size_t index;
  for (index = 0; index < sizeof(names) / sizeof(names[0]); ++index) {
    if ((flags & names[index].flag) != 0u) {
      if (fprintf(stream, "%s\"%s\"", first ? "" : ", ", names[index].name) < 0) {
        return 0;
      }
      first = false;
    }
  }
  return 1;
}

static int write_comparison_report(FILE *stream, const char *baseline_path,
                                   const file_buffer *baseline, const char *candidate_path,
                                   const file_buffer *candidate,
                                   const npunlock_graph_routing_comparison *comparison) {
  char baseline_hash[65];
  char candidate_hash[65];
  size_t index;
  bool accepted = comparison->contract_stable && comparison->unique_changed_group;
  hash_view((npunlock_view){baseline->data, baseline->size}, baseline_hash);
  hash_view((npunlock_view){candidate->data, candidate->size}, candidate_hash);
  if (fprintf(stream,
              "{\n  \"schema\": \"npunlock.porting.compare.v1\",\n  \"baseline\": {\"path\": ") <
          0 ||
      !write_json_string(stream, baseline_path) ||
      fprintf(stream, ", \"size\": %zu, \"sha256\": \"%s\"},\n", baseline->size, baseline_hash) <
          0 ||
      fprintf(stream, "  \"candidate\": {\"path\": ") < 0 ||
      !write_json_string(stream, candidate_path) ||
      fprintf(stream, ", \"size\": %zu, \"sha256\": \"%s\"},\n", candidate->size, candidate_hash) <
          0 ||
      fprintf(stream,
              "  \"comparison\": {\"contract_stable\": %s, "
              "\"unique_changed_group\": %s, \"accepted\": %s, "
              "\"mismatch_flags\": %u, \"mismatch_names\": [",
              comparison->contract_stable ? "true" : "false",
              comparison->unique_changed_group ? "true" : "false", accepted ? "true" : "false",
              comparison->mismatch_flags) < 0 ||
      !write_mismatch_names(stream, comparison->mismatch_flags) ||
      fprintf(stream, "], \"changed_group_indices\": [") < 0) {
    return 0;
  }
  for (index = 0; index < comparison->changed_group_count; ++index) {
    if (fprintf(stream, "%s%u", index == 0 ? "" : ", ", comparison->changed_group_indices[index]) <
        0) {
      return 0;
    }
  }
  return fprintf(stream,
                 "]},\n  \"scope\": \"diagnostic comparison of validated NPU3720/compiler-8.3 "
                 "native observations; code offsets and range indices may move, but every tensor, "
                 "routing, scheduling, output, DPU, and derived relation must remain equal; "
                 "accepted means one code group changed under a stable native contract, not that "
                 "the group has a proven symbolic source identity\"\n}\n") >= 0;
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
  const char *compare_path = NULL;
  const char *report_path = NULL;
  file_buffer graph = {0};
  file_buffer compare_graph = {0};
  npunlock_graph_routing_observations routing = {0};
  npunlock_graph_routing_observations compare_routing = {0};
  npunlock_graph_routing_comparison comparison = {0};
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
    } else if (strcmp(argv[index], "--compare") == 0 && index + 1 < argc && compare_path == NULL) {
      compare_path = argv[++index];
    } else if (strcmp(argv[index], "--report") == 0 && index + 1 < argc && report_path == NULL) {
      report_path = argv[++index];
    } else {
      fprintf(stderr, "unknown or incomplete option: %s\n", argv[index]);
      print_usage(argv[0]);
      return 2;
    }
  }
  if (graph_path == NULL ||
      (report_path != NULL && (strcmp(graph_path, report_path) == 0 ||
                               (compare_path != NULL && strcmp(compare_path, report_path) == 0)))) {
    fprintf(stderr, "--graph is required and the report path must be distinct from inputs\n");
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
  if (compare_path != NULL) {
    compare_graph = read_file(compare_path);
    if (compare_graph.size == SIZE_MAX) {
      goto cleanup;
    }
    status = npunlock_inspect_graph_routing((npunlock_view){compare_graph.data, compare_graph.size},
                                            &compare_routing, &diagnostic);
    if (status != NPUNLOCK_STATUS_OK) {
      print_diagnostic(&diagnostic);
      goto cleanup;
    }
    status = npunlock_compare_graph_routing_observations(&routing, &compare_routing, &comparison,
                                                         &diagnostic);
    if (status != NPUNLOCK_STATUS_OK) {
      print_diagnostic(&diagnostic);
      goto cleanup;
    }
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
  if (!(compare_path == NULL ? write_report(report, graph_path, &graph, &routing)
                             : write_comparison_report(report, graph_path, &graph, compare_path,
                                                       &compare_graph, &comparison)) ||
      fflush(report) != 0) {
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
  npunlock_graph_routing_observations_release(&compare_routing);
  npunlock_graph_routing_comparison_release(&comparison);
  npunlock_diagnostic_release(&diagnostic);
  free(graph.data);
  free(compare_graph.data);
  return success ? 0 : 1;
}
