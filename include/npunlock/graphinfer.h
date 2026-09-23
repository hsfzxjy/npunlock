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

/* Opaque in-process graph execution state. */
typedef struct graphinfer_session graphinfer_session;

typedef struct graphinfer_session_result {
  uint32_t struct_size;
  uint32_t selected_driver_index;
  uint32_t selected_device_index;
  uint32_t driver_version;
  uint32_t device_vendor_id;
  uint32_t device_id;
  graphinfer_session *session;
  npunlock_diagnostic diagnostic;
} graphinfer_session_result;

/*
 * A host-visible Level Zero shared allocation. data remains valid until
 * graphinfer_shared_buffer_release(). The private implementation field is not
 * part of the caller contract and must not be modified. This is an owning
 * object: do not copy it or release a copy.
 */
typedef struct graphinfer_shared_buffer {
  uint32_t struct_size;
  uint8_t *data;
  size_t size;
  void *implementation;
  npunlock_diagnostic diagnostic;
} graphinfer_shared_buffer;

typedef struct graphinfer_shared_tensor {
  uint32_t struct_size;
  uint32_t argument_index;
  npunlock_view argument_name_utf8;
  graphinfer_shared_buffer *buffer;
} graphinfer_shared_tensor;

typedef struct graphinfer_session_infer_result {
  uint32_t struct_size;
  npunlock_diagnostic diagnostic;
} graphinfer_session_infer_result;

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

/*
 * Create an in-process graph session. Unlike graphinfer_infer(), this path is
 * not isolated in a killable worker because the NPU and caller must share the
 * same Level Zero allocations. timeout_ms still bounds submitted fence waits,
 * but cannot terminate a driver call that never returns.
 * worker_executable_utf8 is ignored by this entry point.
 */
NPUNLOCK_GRAPHINFER_API npunlock_status graphinfer_session_create(
    const graphinfer_options *options, npunlock_view graph_blob, graphinfer_session_result *result);

/*
 * Close the public session and release its graph. Existing shared buffers keep
 * the underlying Level Zero context alive until their own release functions
 * are called, but cannot be used for further inference after session close.
 */
NPUNLOCK_GRAPHINFER_API void graphinfer_session_result_release(graphinfer_session_result *result);

NPUNLOCK_GRAPHINFER_API npunlock_status graphinfer_shared_buffer_create(
    graphinfer_session *session, size_t size, graphinfer_shared_buffer *buffer);

NPUNLOCK_GRAPHINFER_API void graphinfer_shared_buffer_release(graphinfer_shared_buffer *buffer);

/*
 * inputs and outputs must exactly cover the graph arguments. Every buffer must
 * belong to session and have the exact byte size required by its argument.
 * Buffers are borrowed for this synchronous call. Calls on one session are
 * serialized internally.
 */
NPUNLOCK_GRAPHINFER_API npunlock_status
graphinfer_session_infer(graphinfer_session *session, const graphinfer_shared_tensor *inputs,
                         size_t input_count, const graphinfer_shared_tensor *outputs,
                         size_t output_count, graphinfer_session_infer_result *result);

NPUNLOCK_GRAPHINFER_API void
graphinfer_session_infer_result_release(graphinfer_session_infer_result *result);

#ifdef __cplusplus
}
#endif

#endif
