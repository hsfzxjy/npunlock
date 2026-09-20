#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "npunlock/patchblob.h"

#define CHECK(expression)                                                                          \
  do {                                                                                             \
    if (!(expression)) {                                                                           \
      fprintf(stderr, "check failed at %s:%d: %s\n", __FILE__, __LINE__, #expression);             \
      return 1;                                                                                    \
    }                                                                                              \
  } while (0)

static npunlock_buffer read_file(const char *path) {
  npunlock_buffer buffer = {0};
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
    return buffer;
  }
  length = ftell(stream);
  if (length <= 0 || fseek(stream, 0, SEEK_SET) != 0) {
    fclose(stream);
    return buffer;
  }
  buffer.data = (uint8_t *)malloc((size_t)length);
  if (buffer.data == NULL || fread(buffer.data, 1, (size_t)length, stream) != (size_t)length) {
    free(buffer.data);
    buffer.data = NULL;
    fclose(stream);
    return buffer;
  }
  buffer.size = (size_t)length;
  fclose(stream);
  return buffer;
}

int main(int argument_count, char **arguments) {
  npunlock_buffer carrier;
  npunlock_buffer elf;
  npunlock_buffer expected;
  patchblob_options options = {0};
  patchblob_target targets[2] = {{0}};
  patchblob_result result = {0};
  uint32_t flags = PATCHBLOB_CONTRACT_STATIC | PATCHBLOB_CONTRACT_DENSE | PATCHBLOB_CONTRACT_FP16 |
                   PATCHBLOB_CONTRACT_CMX | PATCHBLOB_CONTRACT_DISJOINT_OUTPUT;
  size_t index;
  npunlock_status status;

  CHECK(argument_count == 4);
  carrier = read_file(arguments[1]);
  elf = read_file(arguments[2]);
  expected = read_file(arguments[3]);
  CHECK(carrier.data != NULL && elf.data != NULL && expected.data != NULL);
  options.struct_size = sizeof(options);
  options.image_alignment = 0x400;
  options.tail_padding = 0x80;
  for (index = 0; index < 2; ++index) {
    targets[index].struct_size = sizeof(targets[index]);
    targets[index].invocation_index = (uint32_t)index;
    targets[index].range_index = (uint32_t)index;
    targets[index].expected_input_count = 1;
    targets[index].expected_element_count = 8;
    targets[index].expected_span_bytes = 16;
    targets[index].required_contract_flags = flags;
  }
  status = patchblob_patch(&options, (npunlock_view){carrier.data, carrier.size},
                           (npunlock_view){elf.data, elf.size}, targets, 2, &result);
  if (status != NPUNLOCK_STATUS_OK && result.diagnostic.json.data != NULL) {
    fprintf(stderr, "%s\n", (const char *)result.diagnostic.json.data);
  }
  CHECK(status == NPUNLOCK_STATUS_OK);
  CHECK(result.graph_blob.size == expected.size);
  CHECK(memcmp(result.graph_blob.data, expected.data, expected.size) == 0);
  CHECK(result.report_json.data != NULL);
  CHECK(strstr((const char *)result.report_json.data, "npunlock.patchblob.v1") != NULL);
  CHECK(strstr((const char *)result.report_json.data, "\"invocation_index\":1") != NULL);
  patchblob_result_release(&result);
  free(expected.data);
  free(elf.data);
  free(carrier.data);
  return 0;
}
