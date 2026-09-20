#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "npunlock/ir2blob.h"
#include "npunlock/patchblob.h"
#include "npunlock/shavecc.h"

#define CHECK(expression)                                                                          \
  do {                                                                                             \
    if (!(expression)) {                                                                           \
      fprintf(stderr, "check failed at %s:%d: %s\n", __FILE__, __LINE__, #expression);             \
      return 1;                                                                                    \
    }                                                                                              \
  } while (0)

static int check_diagnostic(const npunlock_diagnostic *diagnostic, const char *status) {
  return diagnostic->struct_size == sizeof(*diagnostic) && diagnostic->json.data != NULL &&
         diagnostic->json.size != 0 && strstr((const char *)diagnostic->json.data, status) != NULL;
}

int main(void) {
  static const uint8_t source[] = "void controlled_act(unsigned p) {(void)p;}";
  static const uint8_t path[] = "C:\\npunlock-tests\\missing-movitools";
  static const uint8_t missing_ir_worker[] = "C:\\npunlock-tests\\missing-ir-worker.exe";
  static const uint8_t cpu[] = "3720xx";
  static const uint8_t entry[] = "controlled_act";
  static const uint8_t script[] = "SECTIONS {}";
  static const uint8_t xml[] = "<net/>";
  static const uint8_t blob[] = {0x7f, 'E', 'L', 'F'};
  shavecc_options shave_options = {0};
  shavecc_result shave_result = {0};
  ir2blob_options ir_options = {0};
  ir2blob_result ir_result = {0};
  patchblob_options patch_options = {0};
  patchblob_target target = {0};
  patchblob_result patch_result = {0};

  shave_options.struct_size = sizeof(shave_options);
  shave_options.movi_dll_directory_utf8 = (npunlock_view){path, sizeof(path) - 1};
  shave_options.target_cpu = (npunlock_view){cpu, sizeof(cpu) - 1};
  shave_options.entry_symbol = (npunlock_view){entry, sizeof(entry) - 1};
  shave_options.linker_script = (npunlock_view){script, sizeof(script) - 1};
  shave_options.timeout_ms = 1000;
  CHECK(shavecc_compile(&shave_options, (npunlock_view){source, sizeof(source) - 1},
                        &shave_result) == NPUNLOCK_STATUS_NOT_FOUND);
  CHECK(check_diagnostic(&shave_result.diagnostic, "not_found"));
  shavecc_result_release(&shave_result);
  shavecc_result_release(&shave_result);

  ir_options.struct_size = sizeof(ir_options);
  ir_options.driver_index = IR2BLOB_AUTO_INDEX;
  ir_options.device_index = IR2BLOB_AUTO_INDEX;
  ir_options.timeout_ms = 1000;
  ir_options.worker_executable_utf8 =
      (npunlock_view){missing_ir_worker, sizeof(missing_ir_worker) - 1};
  CHECK(ir2blob_compile(&ir_options, (npunlock_view){xml, sizeof(xml) - 1},
                        (npunlock_view){NULL, 0}, &ir_result) == NPUNLOCK_STATUS_NOT_FOUND);
  CHECK(check_diagnostic(&ir_result.diagnostic, "not_found"));
  ir2blob_result_release(&ir_result);

  patch_options.struct_size = sizeof(patch_options);
  patch_options.image_alignment = 0x400;
  patch_options.tail_padding = 0x80;
  target.struct_size = sizeof(target);
  target.invocation_index = 0;
  target.range_index = 0;
  target.expected_input_count = 1;
  target.expected_element_count = 16;
  target.expected_span_bytes = 32;
  target.required_contract_flags = PATCHBLOB_CONTRACT_STATIC | PATCHBLOB_CONTRACT_DENSE |
                                   PATCHBLOB_CONTRACT_FP16 | PATCHBLOB_CONTRACT_CMX |
                                   PATCHBLOB_CONTRACT_DISJOINT_OUTPUT;
  CHECK(patchblob_patch(&patch_options, (npunlock_view){blob, sizeof(blob)},
                        (npunlock_view){blob, sizeof(blob)}, &target, 1,
                        &patch_result) == NPUNLOCK_STATUS_MALFORMED_INPUT);
  CHECK(check_diagnostic(&patch_result.diagnostic, "malformed_input"));
  patchblob_result_release(&patch_result);

  CHECK(shavecc_compile(NULL, (npunlock_view){source, sizeof(source) - 1}, &shave_result) ==
        NPUNLOCK_STATUS_INVALID_ARGUMENT);
  CHECK(check_diagnostic(&shave_result.diagnostic, "invalid_argument"));
  shavecc_result_release(&shave_result);
  return 0;
}
