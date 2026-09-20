#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "npunlock/graphinfer.h"

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
  if (length <= 0 || (uint64_t)length > SIZE_MAX || length > 512 * 1024 * 1024 ||
      _fseeki64(file, 0, SEEK_SET) != 0) {
    fclose(file);
    return 0;
  }
  buffer->data = (uint8_t *)malloc((size_t)length);
  if (buffer->data == NULL) {
    fclose(file);
    return 0;
  }
  received = fread(buffer->data, 1, (size_t)length, file);
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
  file_buffer graph = {0};
  file_buffer input = {0};
  file_buffer expected = {0};
  graphinfer_options options = {0};
  graphinfer_input graph_input = {0};
  graphinfer_result result = {0};
  npunlock_status status;
  int return_code = 1;
  if (argc != 4 || !read_file(argv[1], &graph) || !read_file(argv[2], &input) ||
      !read_file(argv[3], &expected)) {
    fprintf(stderr, "usage: %s NATIVE_GRAPH INPUT EXPECTED_OUTPUT\n", argv[0]);
    goto done;
  }
  options.struct_size = sizeof(options);
  options.driver_index = GRAPHINFER_AUTO_INDEX;
  options.device_index = GRAPHINFER_AUTO_INDEX;
  options.timeout_ms = 15000;
  graph_input.struct_size = sizeof(graph_input);
  graph_input.argument_index = 0;
  graph_input.data = (npunlock_view){input.data, input.size};
  status =
      graphinfer_infer(&options, (npunlock_view){graph.data, graph.size}, &graph_input, 1, &result);
  if (status != NPUNLOCK_STATUS_OK) {
    fprintf(stderr, "graphinfer_infer failed: %s\n", npunlock_status_name(status));
    if (result.diagnostic.json.data != NULL) {
      fwrite(result.diagnostic.json.data, 1, result.diagnostic.json.size, stderr);
    }
    goto done;
  }
  if (result.device_vendor_id != 0x8086 || result.output_count != 1 ||
      result.outputs[0].precision != GRAPHINFER_PRECISION_FP16 ||
      result.outputs[0].data.size != expected.size ||
      memcmp(result.outputs[0].data.data, expected.data, expected.size) != 0) {
    fprintf(stderr, "unexpected inference result: vendor=0x%04x outputs=%zu\n",
            result.device_vendor_id, result.output_count);
    goto done;
  }
  printf("executed caller graph and tensor: %zu input bytes, %zu output bytes; driver=0x%08x\n",
         input.size, result.outputs[0].data.size, result.driver_version);
  return_code = 0;

done:
  graphinfer_result_release(&result);
  release_file(&expected);
  release_file(&input);
  release_file(&graph);
  return return_code;
}
