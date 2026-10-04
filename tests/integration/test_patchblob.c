#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "npunlock/patchblob.h"

#define FIXTURE_ROOT "tests/fixtures/npu3720/"

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

int main(void) {
  npunlock_buffer carrier;
  npunlock_buffer two_operation_carrier;
  npunlock_buffer broadcast_carrier;
  npunlock_buffer convert_f32_to_f16;
  npunlock_buffer convert_f16_to_f32;
  npunlock_buffer elf;
  npunlock_buffer expected;
  patchblob_options options = {0};
  patchblob_target targets[2] = {{0}};
  patchblob_target broadcast_targets[4] = {{0}};
  patchblob_target_v2 conversion_targets[2] = {{0}};
  patchblob_discovery_result discovery = {0};
  patchblob_discovery_result_v2 discovery_v2 = {0};
  patchblob_result result = {0};
  uint32_t flags = PATCHBLOB_CONTRACT_STATIC | PATCHBLOB_CONTRACT_DENSE | PATCHBLOB_CONTRACT_FP16 |
                   PATCHBLOB_CONTRACT_CMX | PATCHBLOB_CONTRACT_DISJOINT_OUTPUT;
  size_t index;
  npunlock_status status;

  carrier = read_file(FIXTURE_ROOT "abs-add-1x16-tile1.blob");
  two_operation_carrier = read_file(FIXTURE_ROOT "abs-abs-1x32.blob");
  broadcast_carrier = read_file(FIXTURE_ROOT "maximum-broadcast-1x32-f16.blob");
  convert_f32_to_f16 = read_file(FIXTURE_ROOT "convert-f32-to-f16-1x16.blob");
  convert_f16_to_f32 = read_file(FIXTURE_ROOT "convert-f16-to-f32-1x16.blob");
  elf = read_file(FIXTURE_ROOT "add1-fp16.elf");
  expected = read_file(FIXTURE_ROOT "add1-shared-1x16.blob");
  CHECK(carrier.data != NULL && two_operation_carrier.data != NULL &&
        broadcast_carrier.data != NULL && convert_f32_to_f16.data != NULL &&
        convert_f16_to_f32.data != NULL && elf.data != NULL && expected.data != NULL);
  status = patchblob_discover_targets((npunlock_view){carrier.data, carrier.size}, &discovery);
  if (status != NPUNLOCK_STATUS_OK && discovery.diagnostic.json.data != NULL) {
    fprintf(stderr, "%s\n", (const char *)discovery.diagnostic.json.data);
  }
  CHECK(status == NPUNLOCK_STATUS_OK);
  CHECK(discovery.group_count == 1);
  CHECK(discovery.target_count == 2);
  for (index = 0; index < discovery.target_count; ++index) {
    CHECK(discovery.targets[index].group_index == 0);
    CHECK(discovery.targets[index].target.invocation_index == index);
    CHECK(discovery.targets[index].target.range_index == index);
    CHECK(discovery.targets[index].target.expected_input_count == 1);
    CHECK(discovery.targets[index].target.expected_element_count == 8);
    CHECK(discovery.targets[index].target.expected_span_bytes == 16);
  }
  patchblob_discovery_result_release(&discovery);
  status = patchblob_discover_targets(
      (npunlock_view){convert_f32_to_f16.data, convert_f32_to_f16.size}, &discovery);
  CHECK(status == NPUNLOCK_STATUS_UNSUPPORTED);
  patchblob_discovery_result_release(&discovery);
  status = patchblob_discover_targets_v2(
      (npunlock_view){convert_f32_to_f16.data, convert_f32_to_f16.size}, &discovery_v2);
  if (status != NPUNLOCK_STATUS_OK && discovery_v2.diagnostic.json.data != NULL) {
    fprintf(stderr, "%s\n", (const char *)discovery_v2.diagnostic.json.data);
  }
  CHECK(status == NPUNLOCK_STATUS_OK);
  CHECK(discovery_v2.group_count == 2);
  CHECK(discovery_v2.target_count == 6);
  for (index = 0; index < 2; ++index) {
    const patchblob_target_v2 *target = &discovery_v2.targets[index].target;
    CHECK(discovery_v2.targets[index].group_index == 0);
    CHECK(target->target_flags == PATCHBLOB_TARGET_UNARY_CONVERSION_V2);
    CHECK(target->tensor_count == 2);
    CHECK(target->tensors[0].role == PATCHBLOB_TENSOR_ROLE_INPUT_V2);
    CHECK(target->tensors[0].precision == PATCHBLOB_TENSOR_PRECISION_FP32_V2);
    CHECK(target->tensors[0].element_count == 16);
    CHECK(target->tensors[0].span_bytes == 64);
    CHECK(target->tensors[1].role == PATCHBLOB_TENSOR_ROLE_OUTPUT_V2);
    CHECK(target->tensors[1].precision == PATCHBLOB_TENSOR_PRECISION_FP16_V2);
    CHECK(target->tensors[1].element_count == 16);
    CHECK(target->tensors[1].span_bytes == 32);
    conversion_targets[index] = *target;
  }
  options.struct_size = sizeof(options);
  options.image_alignment = 0x400;
  options.tail_padding = 0x80;
  status = patchblob_patch_v2(&options,
                              (npunlock_view){convert_f32_to_f16.data, convert_f32_to_f16.size},
                              (npunlock_view){elf.data, elf.size}, conversion_targets, 1, &result);
  CHECK(status == NPUNLOCK_STATUS_UNSUPPORTED);
  CHECK(strstr((const char *)result.diagnostic.json.data, "complete ACT replica group") != NULL);
  patchblob_result_release(&result);
  status = patchblob_patch_v2(&options,
                              (npunlock_view){convert_f32_to_f16.data, convert_f32_to_f16.size},
                              (npunlock_view){elf.data, elf.size}, conversion_targets, 2, &result);
  CHECK(status == NPUNLOCK_STATUS_OK);
  CHECK(result.graph_blob.size > convert_f32_to_f16.size);
  CHECK(strstr((const char *)result.report_json.data, "npunlock.patchblob.v2") != NULL);
  CHECK(strstr((const char *)result.report_json.data, "\"contract_version\":2") != NULL);
  CHECK(strstr((const char *)result.report_json.data, "\"precision\":\"fp32\"") != NULL);
  CHECK(strstr((const char *)result.report_json.data, "\"precision\":\"fp16\"") != NULL);
  patchblob_result_release(&result);
  patchblob_discovery_result_v2_release(&discovery_v2);
  status = patchblob_discover_targets_v2(
      (npunlock_view){convert_f16_to_f32.data, convert_f16_to_f32.size}, &discovery_v2);
  CHECK(status == NPUNLOCK_STATUS_OK);
  CHECK(discovery_v2.group_count == 2);
  CHECK(discovery_v2.target_count == 6);
  for (index = 4; index < 6; ++index) {
    const patchblob_target_v2 *target = &discovery_v2.targets[index].target;
    CHECK(discovery_v2.targets[index].group_index == 1);
    CHECK(target->target_flags == PATCHBLOB_TARGET_UNARY_CONVERSION_V2);
    CHECK(target->tensors[0].precision == PATCHBLOB_TENSOR_PRECISION_FP16_V2);
    CHECK(target->tensors[0].span_bytes == 32);
    CHECK(target->tensors[1].precision == PATCHBLOB_TENSOR_PRECISION_FP32_V2);
    CHECK(target->tensors[1].span_bytes == 64);
  }
  patchblob_discovery_result_v2_release(&discovery_v2);
  status = patchblob_discover_targets(
      (npunlock_view){two_operation_carrier.data, two_operation_carrier.size}, &discovery);
  if (status != NPUNLOCK_STATUS_OK && discovery.diagnostic.json.data != NULL) {
    fprintf(stderr, "%s\n", (const char *)discovery.diagnostic.json.data);
  }
  CHECK(status == NPUNLOCK_STATUS_OK);
  CHECK(discovery.group_count == 2);
  CHECK(discovery.target_count == 8);
  for (index = 0; index < discovery.target_count; ++index) {
    CHECK(discovery.targets[index].group_index == index / 4u);
    CHECK(discovery.targets[index].target.invocation_index == index);
    CHECK(discovery.targets[index].target.range_index == index);
    CHECK(discovery.targets[index].target.expected_input_count == 1);
    CHECK(discovery.targets[index].target.expected_element_count == 16);
    CHECK(discovery.targets[index].target.expected_span_bytes == 32);
  }
  patchblob_discovery_result_release(&discovery);
  status = patchblob_discover_targets(
      (npunlock_view){broadcast_carrier.data, broadcast_carrier.size}, &discovery);
  if (status != NPUNLOCK_STATUS_OK && discovery.diagnostic.json.data != NULL) {
    fprintf(stderr, "%s\n", (const char *)discovery.diagnostic.json.data);
  }
  CHECK(status == NPUNLOCK_STATUS_OK);
  CHECK(discovery.group_count == 1);
  CHECK(discovery.target_count == 4);
  for (index = 0; index < discovery.target_count; ++index) {
    const patchblob_target *target = &discovery.targets[index].target;
    CHECK(discovery.targets[index].group_index == 0);
    CHECK(target->invocation_index == index);
    CHECK(target->range_index == index);
    CHECK(target->expected_input_count == 2);
    CHECK(target->expected_element_count == 8);
    CHECK(target->expected_span_bytes == 16);
    CHECK((target->required_contract_flags & PATCHBLOB_CONTRACT_INPUT_1_SCALAR) != 0);
    broadcast_targets[index] = *target;
  }
  patchblob_discovery_result_release(&discovery);
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
  status =
      patchblob_patch(&options, (npunlock_view){broadcast_carrier.data, broadcast_carrier.size},
                      (npunlock_view){elf.data, elf.size}, broadcast_targets, 4, &result);
  if (status != NPUNLOCK_STATUS_OK && result.diagnostic.json.data != NULL) {
    fprintf(stderr, "%s\n", (const char *)result.diagnostic.json.data);
  }
  CHECK(status == NPUNLOCK_STATUS_OK);
  CHECK(result.graph_blob.size > broadcast_carrier.size);
  CHECK(strstr((const char *)result.report_json.data, "\"input_1_scalar\"") != NULL);
  patchblob_result_release(&result);
  broadcast_targets[0].required_contract_flags &= ~PATCHBLOB_CONTRACT_INPUT_1_SCALAR;
  status =
      patchblob_patch(&options, (npunlock_view){broadcast_carrier.data, broadcast_carrier.size},
                      (npunlock_view){elf.data, elf.size}, broadcast_targets, 4, &result);
  CHECK(status == NPUNLOCK_STATUS_UNSUPPORTED);
  CHECK(result.diagnostic.json.data != NULL);
  CHECK(strstr((const char *)result.diagnostic.json.data, "expected element contract") != NULL);
  patchblob_result_release(&result);
  free(expected.data);
  free(elf.data);
  free(broadcast_carrier.data);
  free(convert_f16_to_f32.data);
  free(convert_f32_to_f16.data);
  free(two_operation_carrier.data);
  free(carrier.data);
  return 0;
}
