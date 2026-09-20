#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "npunlock/ir2blob.h"

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
  if (length < 0 || (uint64_t)length > SIZE_MAX || length > 512 * 1024 * 1024 ||
      _fseeki64(file, 0, SEEK_SET) != 0) {
    fclose(file);
    return 0;
  }
  if (length != 0) {
    buffer->data = (uint8_t *)malloc((size_t)length);
    if (buffer->data == NULL) {
      fclose(file);
      return 0;
    }
  }
  received = length == 0 ? 0 : fread(buffer->data, 1, (size_t)length, file);
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
  file_buffer xml = {0};
  file_buffer weights = {0};
  file_buffer expected = {0};
  ir2blob_options options = {0};
  ir2blob_result result = {0};
  npunlock_status status;
  int return_code = 1;
  if (argc != 4 || !read_file(argv[1], &xml) || !read_file(argv[2], &weights) ||
      !read_file(argv[3], &expected)) {
    fprintf(stderr, "usage: %s IR_XML WEIGHTS EXPECTED_NATIVE_BLOB\n", argv[0]);
    goto done;
  }
  options.struct_size = sizeof(options);
  options.driver_index = IR2BLOB_AUTO_INDEX;
  options.device_index = IR2BLOB_AUTO_INDEX;
  options.timeout_ms = 15000;
  status = ir2blob_compile(&options, (npunlock_view){xml.data, xml.size},
                           (npunlock_view){weights.data, weights.size}, &result);
  if (status != NPUNLOCK_STATUS_OK) {
    fprintf(stderr, "ir2blob_compile failed: %s\n", npunlock_status_name(status));
    if (result.diagnostic.json.data != NULL) {
      fwrite(result.diagnostic.json.data, 1, result.diagnostic.json.size, stderr);
    }
    goto done;
  }
  if (result.device_vendor_id != 0x8086 || result.compiler_version_major == 0 ||
      result.graph_blob.size != expected.size ||
      memcmp(result.graph_blob.data, expected.data, expected.size) != 0) {
    fprintf(stderr,
            "unexpected native result: vendor=0x%04x compiler=%u.%u size=%zu expected=%zu\n",
            result.device_vendor_id, result.compiler_version_major, result.compiler_version_minor,
            result.graph_blob.size, expected.size);
    goto done;
  }
  printf("compiled and reloaded native graph: %zu bytes; driver=0x%08x device=%u:%u "
         "pci=%04x:%04x compiler=%u.%u graph-ext=%u.%u\n",
         result.graph_blob.size, result.driver_version, result.selected_driver_index,
         result.selected_device_index, result.device_vendor_id, result.device_id,
         result.compiler_version_major, result.compiler_version_minor,
         result.graph_extension_version >> 16, result.graph_extension_version & 0xffff);
  return_code = 0;

done:
  ir2blob_result_release(&result);
  release_file(&expected);
  release_file(&weights);
  release_file(&xml);
  return return_code;
}
