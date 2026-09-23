#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "npunlock/graphinfer.h"

#define FIXTURE_ROOT "tests/fixtures/npu3720/"

static const uint8_t input_fp16[] = {
    0x00, 0xc0, 0x00, 0xbf, 0x00, 0xbe, 0x00, 0xbd, 0x00, 0xbc, 0x00, 0xba, 0x00, 0xb8, 0x00, 0xb4,
    0x00, 0x00, 0x00, 0x34, 0x00, 0x38, 0x00, 0x3a, 0x00, 0x3c, 0x00, 0x3d, 0x00, 0x3e, 0x00, 0x3f,
};

static const uint8_t expected_fp16[] = {
    0x00, 0xbe, 0xbc, 0xbc, 0xf0, 0xba, 0x66, 0xb8, 0x78, 0xb3, 0x58, 0x2d, 0x66, 0x36, 0xbc, 0x39,
    0x22, 0x3c, 0x66, 0x3d, 0xaa, 0x3e, 0xef, 0x3f, 0x9a, 0x40, 0x3c, 0x41, 0xde, 0x41, 0x80, 0x42,
};

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

int main(void) {
  file_buffer graph = {0};
  graphinfer_options options = {0};
  graphinfer_input graph_input = {0};
  graphinfer_result result = {0};
  graphinfer_session_result session_result = {0};
  graphinfer_shared_buffer shared_input = {0};
  graphinfer_shared_buffer shared_output = {0};
  graphinfer_shared_tensor input_binding = {0};
  graphinfer_shared_tensor output_binding = {0};
  graphinfer_session_infer_result shared_result = {0};
  npunlock_status status;
  uint32_t output_index;
  int return_code = 1;
  if (!read_file(FIXTURE_ROOT "add1-shared-1x16.blob", &graph)) {
    fprintf(stderr, "failed to read bundled graph fixture\n");
    goto done;
  }
  options.struct_size = sizeof(options);
  options.driver_index = GRAPHINFER_AUTO_INDEX;
  options.device_index = GRAPHINFER_AUTO_INDEX;
  options.timeout_ms = 15000;
  graph_input.struct_size = sizeof(graph_input);
  graph_input.argument_index = 0;
  graph_input.data = (npunlock_view){input_fp16, sizeof(input_fp16)};
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
      result.outputs[0].data.size != sizeof(expected_fp16) ||
      memcmp(result.outputs[0].data.data, expected_fp16, sizeof(expected_fp16)) != 0) {
    fprintf(stderr, "unexpected inference result: vendor=0x%04x outputs=%zu\n",
            result.device_vendor_id, result.output_count);
    goto done;
  }
  output_index = result.outputs[0].argument_index;
  status =
      graphinfer_session_create(&options, (npunlock_view){graph.data, graph.size}, &session_result);
  if (status != NPUNLOCK_STATUS_OK || session_result.session == NULL) {
    fprintf(stderr, "graphinfer_session_create failed: %s\n", npunlock_status_name(status));
    goto done;
  }
  status =
      graphinfer_shared_buffer_create(session_result.session, sizeof(input_fp16), &shared_input);
  if (status == NPUNLOCK_STATUS_OK) {
    status = graphinfer_shared_buffer_create(session_result.session, sizeof(expected_fp16),
                                             &shared_output);
  }
  if (status != NPUNLOCK_STATUS_OK) {
    fprintf(stderr, "shared buffer creation failed: %s\n", npunlock_status_name(status));
    goto done;
  }
  memcpy(shared_input.data, input_fp16, sizeof(input_fp16));
  input_binding.struct_size = sizeof(input_binding);
  input_binding.argument_index = 0;
  input_binding.buffer = &shared_input;
  output_binding.struct_size = sizeof(output_binding);
  output_binding.argument_index = output_index;
  output_binding.buffer = &shared_output;
  status = graphinfer_session_infer(session_result.session, &input_binding, 1, &output_binding, 1,
                                    &shared_result);
  if (status != NPUNLOCK_STATUS_OK ||
      memcmp(shared_output.data, expected_fp16, sizeof(expected_fp16)) != 0) {
    fprintf(stderr, "shared graph inference failed: %s\n", npunlock_status_name(status));
    goto done;
  }
  graphinfer_session_result_release(&session_result);
  if (memcmp(shared_output.data, expected_fp16, sizeof(expected_fp16)) != 0) {
    fprintf(stderr, "shared buffer did not survive public session release\n");
    goto done;
  }
  printf("executed caller graph and tensor: %zu input bytes, %zu output bytes; driver=0x%08x\n",
         sizeof(input_fp16), result.outputs[0].data.size, result.driver_version);
  return_code = 0;

done:
  graphinfer_session_infer_result_release(&shared_result);
  graphinfer_shared_buffer_release(&shared_output);
  graphinfer_shared_buffer_release(&shared_input);
  graphinfer_session_result_release(&session_result);
  graphinfer_result_release(&result);
  release_file(&graph);
  return return_code;
}
