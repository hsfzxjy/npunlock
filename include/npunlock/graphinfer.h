#ifndef NPUNLOCK_GRAPHINFER_H
#define NPUNLOCK_GRAPHINFER_H

#include <stddef.h>
#include <stdint.h>

#include "npunlock/buffer.h"
#include "npunlock/error.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GRAPHINFER_AUTO_INDEX UINT32_MAX
#define GRAPHINFER_MAX_DIMS 5u

typedef enum graphinfer_precision {
  GRAPHINFER_PRECISION_FP32 = 1,
  GRAPHINFER_PRECISION_FP16 = 2
} graphinfer_precision;

typedef struct graphinfer_options {
  uint32_t struct_size;
  /* Use GRAPHINFER_AUTO_INDEX to select the first matching Intel VPU. */
  uint32_t driver_index;
  uint32_t device_index;
  /* Finite parent-process deadline; zero is invalid. */
  uint32_t timeout_ms;
  /* Empty selects the installed sibling npunlock_worker executable. */
  npunlock_view worker_executable_utf8;
} graphinfer_options;

/*
 * Select an input either by argument_index, or by setting argument_index to
 * GRAPHINFER_AUTO_INDEX and providing argument_name_utf8. The data view is
 * borrowed for the duration of graphinfer_infer().
 */
typedef struct graphinfer_input {
  uint32_t struct_size;
  uint32_t argument_index;
  npunlock_view argument_name_utf8;
  npunlock_view data;
} graphinfer_input;

/* All buffers in an output are owned by the enclosing result. */
typedef struct graphinfer_output {
  uint32_t struct_size;
  uint32_t argument_index;
  uint32_t precision;
  uint32_t dims_count;
  uint32_t dims[GRAPHINFER_MAX_DIMS];
  npunlock_buffer argument_name_utf8;
  npunlock_buffer data;
} graphinfer_output;

typedef struct graphinfer_result {
  uint32_t struct_size;
  uint32_t selected_driver_index;
  uint32_t selected_device_index;
  uint32_t driver_version;
  uint32_t device_vendor_id;
  uint32_t device_id;
  graphinfer_output *outputs;
  size_t output_count;
  /* Verbatim output captured from the bounded execution worker. */
  npunlock_buffer stdout_log;
  npunlock_buffer stderr_log;
  npunlock_diagnostic diagnostic;
} graphinfer_result;

/*
 * graph_blob and every input view are borrowed only for this synchronous call.
 * On success, outputs contains every graph output. On any return after result
 * validation, graphinfer_result_release() is safe and releases partial state.
 * Calls use independent worker processes and do not share execution state.
 */
NPUNLOCK_GRAPHINFER_API npunlock_status graphinfer_infer(const graphinfer_options *options,
                                                         npunlock_view graph_blob,
                                                         const graphinfer_input *inputs,
                                                         size_t input_count,
                                                         graphinfer_result *result);

NPUNLOCK_GRAPHINFER_API void graphinfer_result_release(graphinfer_result *result);

#ifdef __cplusplus
}
#endif

#endif
