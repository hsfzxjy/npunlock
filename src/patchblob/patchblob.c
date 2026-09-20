#include "npunlock/patchblob.h"

#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "elf32.h"
#include "graph_blob.h"
#include "internal.h"

typedef struct report_writer {
  char *data;
  size_t size;
  size_t capacity;
} report_writer;

static bool report_append(report_writer *writer, const char *format, ...) {
  va_list arguments;
  int written;
  size_t available;
  if (writer->size >= writer->capacity) {
    return false;
  }
  available = writer->capacity - writer->size;
  va_start(arguments, format);
  written = vsnprintf(writer->data + writer->size, available, format, arguments);
  va_end(arguments);
  if (written < 0 || (size_t)written >= available) {
    return false;
  }
  writer->size += (size_t)written;
  return true;
}

static void digest_hex(const uint8_t digest[32], char text[65]) {
  static const char digits[] = "0123456789abcdef";
  size_t index;
  for (index = 0; index < 32; ++index) {
    text[index * 2] = digits[digest[index] >> 4];
    text[index * 2 + 1] = digits[digest[index] & 15u];
  }
  text[64] = 0;
}

static npunlock_status build_report(npunlock_view graph_blob, npunlock_view shave_elf,
                                    npunlock_view output_blob,
                                    const npunlock_patch_summary *summary, npunlock_buffer *report,
                                    npunlock_diagnostic *diagnostic) {
  uint8_t graph_digest[32];
  uint8_t elf_digest[32];
  uint8_t output_digest[32];
  char graph_hash[65];
  char elf_hash[65];
  char output_hash[65];
  size_t target_space;
  size_t capacity;
  report_writer writer;
  size_t index;

  if (!npunlock_checked_mul_size(summary->detail_count, 640u, &target_space) ||
      !npunlock_checked_add_size(2048u, target_space, &capacity)) {
    return npunlock_set_diagnostic(diagnostic, NPUNLOCK_STATUS_OVERFLOW, "patchblob.report",
                                   "patch report size overflows");
  }
  writer.data = (char *)malloc(capacity);
  if (writer.data == NULL) {
    return npunlock_set_diagnostic(diagnostic, NPUNLOCK_STATUS_OUT_OF_MEMORY, "patchblob.report",
                                   "could not allocate patch report");
  }
  writer.size = 0;
  writer.capacity = capacity;
  npunlock_sha256(graph_blob, graph_digest);
  npunlock_sha256(shave_elf, elf_digest);
  npunlock_sha256(output_blob, output_digest);
  digest_hex(graph_digest, graph_hash);
  digest_hex(elf_digest, elf_hash);
  digest_hex(output_digest, output_hash);
  if (!report_append(&writer,
                     "{\"schema\":\"npunlock.patchblob.v1\","
                     "\"input_graph_sha256\":\"%s\",\"shave_elf_sha256\":\"%s\","
                     "\"output_graph_sha256\":\"%s\","
                     "\"append\":{\"file_offset\":%zu,\"inserted_size\":%zu,"
                     "\"image_base\":%zu,\"image_size\":%zu},"
                     "\"section_table\":{\"old_offset\":%zu,\"new_offset\":%zu},"
                     "\"validation\":{\"section_contents_preserved\":true,"
                     "\"mutation_set_preserved\":true},\"targets\":[",
                     graph_hash, elf_hash, output_hash, summary->insertion_file_offset,
                     summary->inserted_size, summary->image_base, summary->image_size,
                     summary->old_section_table_offset, summary->new_section_table_offset)) {
    free(writer.data);
    return npunlock_set_diagnostic(diagnostic, NPUNLOCK_STATUS_INTERNAL_ERROR, "patchblob.report",
                                   "patch report buffer was too small");
  }
  for (index = 0; index < summary->detail_count; ++index) {
    const npunlock_patch_detail *detail = &summary->details[index];
    if (!report_append(&writer,
                       "%s{\"invocation_index\":%u,\"range_index\":%u,\"input_count\":%u,"
                       "\"parameter_base\":%" PRIu64 ",\"element_count\":%" PRIu64
                       ",\"span_bytes\":%" PRIu64 ",\"extent\":{\"file_offset\":%zu,"
                       "\"old\":%u,\"new\":%u},\"code_relocation_addend\":{"
                       "\"file_offset\":%zu,\"old\":%" PRId64 ",\"new\":%" PRId64
                       "},\"contract\":[\"static\",\"dense\",\"fp16\",\"cmx\","
                       "\"disjoint_output\"]}",
                       index == 0 ? "" : ",", detail->invocation_index, detail->range_index,
                       detail->input_count, detail->parameter_base, detail->element_count,
                       detail->span_bytes, detail->extent_file_offset, detail->old_extent,
                       detail->new_extent, detail->addend_file_offset, detail->old_addend,
                       detail->new_addend)) {
      free(writer.data);
      return npunlock_set_diagnostic(diagnostic, NPUNLOCK_STATUS_INTERNAL_ERROR, "patchblob.report",
                                     "patch report buffer was too small");
    }
  }
  if (!report_append(&writer, "]}\n")) {
    free(writer.data);
    return npunlock_set_diagnostic(diagnostic, NPUNLOCK_STATUS_INTERNAL_ERROR, "patchblob.report",
                                   "patch report buffer was too small");
  }
  return npunlock_buffer_adopt_malloc((uint8_t *)writer.data, writer.size + 1u, report);
}

npunlock_status patchblob_patch(const patchblob_options *options, npunlock_view graph_blob,
                                npunlock_view shave_elf, const patchblob_target *targets,
                                size_t target_count, patchblob_result *result) {
  npunlock_shave_image shave_image;
  npunlock_patch_summary summary;
  npunlock_buffer patched = {0};
  npunlock_buffer report = {0};
  npunlock_status status;
  size_t index;
  if (result == NULL) {
    return NPUNLOCK_STATUS_INVALID_ARGUMENT;
  }
  memset(result, 0, sizeof(*result));
  result->struct_size = (uint32_t)sizeof(*result);
  memset(&summary, 0, sizeof(summary));
  if (options == NULL || options->struct_size < sizeof(*options) ||
      !npunlock_view_is_valid(graph_blob) || graph_blob.size == 0 ||
      !npunlock_view_is_valid(shave_elf) || shave_elf.size == 0 || targets == NULL ||
      target_count == 0 || options->image_alignment == 0 ||
      (options->image_alignment & (options->image_alignment - 1)) != 0) {
    return npunlock_set_diagnostic(&result->diagnostic, NPUNLOCK_STATUS_INVALID_ARGUMENT,
                                   "patchblob.validate",
                                   "invalid options, input views, or target list");
  }
  for (index = 0; index < target_count; ++index) {
    if (targets[index].struct_size < sizeof(targets[index]) ||
        targets[index].invocation_index == PATCHBLOB_UNUSED_INDEX ||
        targets[index].range_index == PATCHBLOB_UNUSED_INDEX ||
        targets[index].expected_input_count == 0 || targets[index].expected_element_count == 0 ||
        targets[index].expected_span_bytes == 0) {
      return npunlock_set_diagnostic(
          &result->diagnostic, NPUNLOCK_STATUS_INVALID_ARGUMENT, "patchblob.validate_target",
          "each target requires an explicit invocation, range, and contract");
    }
  }
  status = npunlock_parse_shave_elf(shave_elf, &shave_image, &result->diagnostic);
  if (status != NPUNLOCK_STATUS_OK) {
    return status;
  }
  status = npunlock_patch_graph_blob(
      graph_blob, (npunlock_view){shave_elf.data + shave_image.file_offset, shave_image.size},
      targets, target_count, options->image_alignment, options->tail_padding, &patched, &summary,
      &result->diagnostic);
  if (status != NPUNLOCK_STATUS_OK) {
    return status;
  }
  status = build_report(graph_blob, shave_elf, (npunlock_view){patched.data, patched.size},
                        &summary, &report, &result->diagnostic);
  npunlock_patch_summary_release(&summary);
  if (status != NPUNLOCK_STATUS_OK) {
    npunlock_buffer_release(&patched);
    return status;
  }
  result->graph_blob = patched;
  result->report_json = report;
  return NPUNLOCK_STATUS_OK;
}

void patchblob_result_release(patchblob_result *result) {
  if (result == NULL) {
    return;
  }
  npunlock_buffer_release(&result->graph_blob);
  npunlock_buffer_release(&result->report_json);
  npunlock_diagnostic_release(&result->diagnostic);
  memset(result, 0, sizeof(*result));
}
