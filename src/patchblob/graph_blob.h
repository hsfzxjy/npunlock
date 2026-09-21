#ifndef NPUNLOCK_PATCHBLOB_GRAPH_BLOB_H
#define NPUNLOCK_PATCHBLOB_GRAPH_BLOB_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "npunlock/patchblob.h"

typedef struct npunlock_patch_detail {
  uint32_t invocation_index;
  uint32_t range_index;
  uint32_t input_count;
  uint64_t parameter_base;
  uint64_t element_count;
  uint64_t span_bytes;
  size_t extent_file_offset;
  size_t addend_file_offset;
  uint32_t old_extent;
  uint32_t new_extent;
  int64_t old_addend;
  int64_t new_addend;
} npunlock_patch_detail;

typedef struct npunlock_patch_summary {
  size_t insertion_file_offset;
  size_t inserted_size;
  size_t image_base;
  size_t image_size;
  size_t old_section_table_offset;
  size_t new_section_table_offset;
  bool section_contents_preserved;
  bool mutation_set_preserved;
  npunlock_patch_detail *details;
  size_t detail_count;
} npunlock_patch_summary;

npunlock_status npunlock_patch_graph_blob(npunlock_view graph_blob, npunlock_view image,
                                          const patchblob_target *targets, size_t target_count,
                                          uint32_t image_alignment, uint32_t tail_padding,
                                          npunlock_buffer *output, npunlock_patch_summary *summary,
                                          npunlock_diagnostic *diagnostic);
void npunlock_patch_summary_release(npunlock_patch_summary *summary);
npunlock_status npunlock_discover_graph_targets(npunlock_view graph_blob,
                                                patchblob_discovered_target **targets,
                                                size_t *target_count, size_t *group_count,
                                                npunlock_diagnostic *diagnostic);

#endif
