#ifndef NPUNLOCK_INFER_ENGINE_H
#define NPUNLOCK_INFER_ENGINE_H

#include <stddef.h>
#include <stdint.h>

#include "infer_protocol.h"

#define NPUNLOCK_INFER_AUTO_INDEX UINT32_MAX

typedef enum npunlock_infer_stage {
  NPUNLOCK_INFER_STAGE_LOADER = 0,
  NPUNLOCK_INFER_STAGE_DEVICE,
  NPUNLOCK_INFER_STAGE_GRAPH_EXTENSION,
  NPUNLOCK_INFER_STAGE_GRAPH_CREATE,
  NPUNLOCK_INFER_STAGE_INITIALIZATION,
  NPUNLOCK_INFER_STAGE_EXECUTION,
  NPUNLOCK_INFER_STAGE_COMPLETE
} npunlock_infer_stage;

typedef struct npunlock_infer_request_input {
  uint32_t argument_index;
  const uint8_t *name;
  size_t name_size;
  const uint8_t *data;
  size_t data_size;
} npunlock_infer_request_input;

typedef struct npunlock_infer_request {
  uint32_t driver_index;
  uint32_t device_index;
  const uint8_t *graph;
  size_t graph_size;
  npunlock_infer_request_input inputs[NPUNLOCK_INFER_MAX_ARGUMENTS];
  size_t input_count;
} npunlock_infer_request;

typedef struct npunlock_infer_result_output {
  uint32_t argument_index;
  uint32_t precision;
  uint32_t dims_count;
  uint32_t dims[5];
  uint8_t name[256];
  size_t name_size;
  uint8_t *data;
  size_t data_size;
} npunlock_infer_result_output;

typedef struct npunlock_infer_result {
  npunlock_infer_worker_status status;
  npunlock_infer_stage stage;
  uint32_t driver_result;
  uint32_t driver_index;
  uint32_t device_index;
  uint32_t driver_version;
  uint32_t vendor_id;
  uint32_t device_id;
  npunlock_infer_result_output outputs[NPUNLOCK_INFER_MAX_ARGUMENTS];
  size_t output_count;
  char *diagnostic;
} npunlock_infer_result;

void npunlock_infer_execute(const npunlock_infer_request *request, npunlock_infer_result *result);

void npunlock_infer_result_release(npunlock_infer_result *result);

const char *npunlock_infer_stage_name(npunlock_infer_stage stage);

#endif
