# C API guide

`npunlock` exposes four independent C17 component APIs from one native runtime
library:

| Library | Input | Output |
| --- | --- | --- |
| `shavecc` | C source and caller-selected MoviTools configuration | validated linked SHAVE ELF |
| `ir2blob` | OpenVINO-format IR XML and weights | native Intel NPU graph blob |
| `patchblob` | native graph, SHAVE ELF, validated ACT targets | discovered targets or patched native graph and JSON report |
| `graphinfer` | native graph and caller tensor buffers | graph output metadata and buffers |

They share `npunlock_view`, `npunlock_buffer`, `npunlock_status`, and
`npunlock_diagnostic`. No library API uses files as transport. On Windows all
symbols are exported by `npunlock.dll`; the component separation remains in
the source/object targets and installed CMake link targets.

## Linking

After installing the CMake package, consumers can link only the stages they
need:

```cmake
find_package(npunlock CONFIG REQUIRED)

add_executable(example example.c)
target_link_libraries(example PRIVATE npunlock::graphinfer)
```

The installed targets are `npunlock::shavecc`, `npunlock::ir2blob`,
`npunlock::patchblob`, and `npunlock::graphinfer`. Each resolves to the same
`npunlock::native` shared runtime, so linking more than one component does not
deploy additional DLLs.

On Windows, deploy `npunlock_worker.exe` beside `npunlock.dll` when using
MoviTools or IR compilation. Those unsafe compiler calls receive separate
finite-lived processes, and callers may provide an explicit worker path in
their options structures. Graph inference does not use the worker. The
installed package places the DLL, worker, and `npurun` in its `bin` directory.

The distributed DLL and worker use the static MSVC runtime. Public views remain
borrowed and returned allocations must still be released through their owning
API. In particular, Python copies returned bytes and calls the DLL release
function; neither Python nor another CRT frees a native allocation.

## Views, buffers, and ownership

`npunlock_view` is borrowed. Its bytes must remain valid only for the duration
of the synchronous call receiving it:

```c
npunlock_view view = {bytes, byte_count};
```

`{NULL, 0}` represents an empty optional view. `{NULL, nonzero}` is invalid.
Required inputs, such as C source, IR XML, graph blobs, and tensor data, must be
nonempty. String views are UTF-8 byte spans and are not required to include a
terminating NUL; embedded NUL bytes are rejected where a string is expected.

`npunlock_buffer` is owned by the library that returned it. Do not call
`free()` on its data or invoke its release callback directly. Release the
enclosing result with its matching function:

```c
shavecc_result_release(&result);
ir2blob_result_release(&result);
patchblob_result_release(&result);
graphinfer_result_release(&result);
```

For reusable host/NPU shared buffers, create an in-process session, allocate
buffers from it, and bind every graph input and output:

```c
graphinfer_session_result session = {0};
graphinfer_shared_buffer input_buffer = {0};
graphinfer_shared_buffer output_buffer = {0};
graphinfer_shared_tensor input_binding = {0};
graphinfer_shared_tensor output_binding = {0};
graphinfer_session_infer_result inference = {0};

status = graphinfer_session_create(&options, graph_blob, &session);
if (status == NPUNLOCK_STATUS_OK) {
  graphinfer_shared_buffer_create(session.session, input_size, &input_buffer);
  graphinfer_shared_buffer_create(session.session, output_size, &output_buffer);
  memcpy(input_buffer.data, input_bytes, input_size);

  input_binding.struct_size = sizeof(input_binding);
  input_binding.argument_index = input_argument_index;
  input_binding.buffer = &input_buffer;
  output_binding.struct_size = sizeof(output_binding);
  output_binding.argument_index = output_argument_index;
  output_binding.buffer = &output_buffer;

  status = graphinfer_session_infer(session.session, &input_binding, 1,
                                    &output_binding, 1, &inference);
  /* Read output_buffer.data after successful synchronous completion. */
}

graphinfer_session_infer_result_release(&inference);
graphinfer_session_result_release(&session);
graphinfer_shared_buffer_release(&output_buffer);
graphinfer_shared_buffer_release(&input_buffer);
```

Each shared buffer owns a reference to its session's Level Zero context, so it
remains valid if the public session result is released first. The graph is
closed at session release and no later inference is allowed; the context is
destroyed after the last shared buffer is released. A
`graphinfer_shared_buffer` is an owning object: do not copy it or release a
copy. Bindings must exactly cover
all graph inputs and outputs, use buffers from the same session, and match each
argument's exact byte size.

Both inference modes are in-process. They use finite fence waits but cannot
forcibly terminate a driver call that never returns.

Result structures should be zero-initialized. Options, target, and input
descriptors carry `struct_size` for ABI validation; calls populate the result's
`struct_size`:

```c
graphinfer_options options = {0};
graphinfer_result result = {0};

options.struct_size = sizeof(options);
```

The public call initializes its result before doing other validation whenever
the result pointer itself is valid. Calling the matching result-release
function is therefore safe after success or failure, and repeated release is
safe.

Diagnostics contain JSON bytes in `diagnostic.json`. Always honor the recorded
size rather than relying on NUL termination:

```c
if (result.diagnostic.json.data != NULL) {
  fwrite(result.diagnostic.json.data, 1, result.diagnostic.json.size, stderr);
}
```

The worker-backed `shavecc_result` and `ir2blob_result` structures own
`stdout_log` and `stderr_log` buffers. These are the worker process's two
operating-system streams, captured separately and available on success or
failure. They are arbitrary byte spans, not quoted JSON and not necessarily
NUL-terminated. The corresponding fields in `graphinfer_result` are retained
for ABI compatibility and remain empty:

```c
if (status != NPUNLOCK_STATUS_OK) {
  fwrite(result.stdout_log.data, 1, result.stdout_log.size, stdout);
  fwrite(result.stderr_log.data, 1, result.stderr_log.size, stderr);
}
```

The matching result-release function releases both streams. The structured
diagnostic remains separate: for example, MoviTools can return compiler errors
through its own diagnostic buffer without writing them to process stderr.

## `shavecc`

`shavecc_compile()` accepts source bytes, an absolute caller-supplied
`MVC_DEPEND` root, target `3720xx`, entry symbol
`controlled_act`, optional `NAME=DECIMAL` compiler definitions, and a finite
timeout. Its three unsafe stages run in separate bounded invocations of the
unified worker. The returned ELF has already passed the project's narrow SHAVE
validation contract.

Kernel source may begin with the exact virtual include
`#include <npunlock/npu3720_kernel.h>`. `shavecc` expands it from embedded bytes
before invoking MoviTools, so the public API remains buffer-only and does not
depend on an include directory. The optional
`#define MLIBM_DEFINE_LINK_COMPAT 1` feature switch may immediately precede the
include. The installed target-scoped header provides metadata, pointer,
tensor-record, and `mlibm.a` link-compatibility helpers.
`shavecc_npu3720_kernel_header()` returns the same immutable library-owned bytes
for inspection or provenance hashing.

An empty `shavecc_options.linker_script` view selects the Apache-2.0 NPU3720
script embedded in the library. A non-empty view is used verbatim as a caller
override. `shavecc_default_linker_script()` returns an immutable, library-owned
view of the embedded bytes for inspection or provenance hashing; callers must
not release or modify it.

The MoviTools binaries are proprietary caller dependencies. They are not
searched for globally or distributed with `npunlock`.
The root contains the three MoviTools DLLs under `bin` and the archives under
`lib`. Passing `bin` itself is not supported. For C sources that reference math
functions, `shavecc` also resolves `lib\mlibm.a` and links it with section
garbage collection; the resulting ELF is still rejected unless its `.arg.data`
is empty.

## `ir2blob`

`ir2blob_compile()` accepts IR XML and weights in memory. Set both selector
fields to `IR2BLOB_AUTO_INDEX` to select the first matching Intel VPU, or supply
explicit indices. An empty build-flags view is valid and is sent to the driver
as a non-null empty string. The result owns the complete exported native graph
blob and records compiler, graph-extension, driver, and device provenance.

This consumes the OpenVINO IR serialization format but does not load, link, or
invoke OpenVINO.

## `patchblob`

`patchblob_patch()` is hardware-free. Every `patchblob_target` identifies an
explicit invocation and range and supplies the expected arity, element count,
span, and required observed contract flags. The library extracts executable
bytes from the validated SHAVE ELF, applies only the confirmed append/rebase
mutation, and returns a new blob plus a JSON preservation report.

Exactly one of `PATCHBLOB_CONTRACT_FP16` and `PATCHBLOB_CONTRACT_FP32` is
required. FP32 support is limited to the validated precision-preserved unary
carrier; it does not make mixed-precision conversion groups patch-compatible.

`patchblob_discover_targets()` validates the graph's supported ACT carriers and
returns owned targets grouped by zero-based positional ACT operation. Release
the result with `patchblob_discovery_result_release()`. Discovery is based on a
narrow observed NPU3720/compiler-8.3 invocation identity and tensor contract;
callers must correlate the group count and order with their own graph before
patching.

Do not use source-level graph node names as patch selectors. Unsupported or
ambiguous graph structures fail closed.

## `graphinfer`

`graphinfer_infer()` executes a caller-provided native graph in-process and
copies tensors through internal Level Zero host allocations. Its
current contract is static FP16 or FP32 graph inputs and outputs
with at most five dimensions. The caller must provide exactly one descriptor
for every graph input, selected either by argument index or by exact UTF-8
name.

```c
graphinfer_options options = {0};
graphinfer_input input = {0};
graphinfer_result result = {0};
npunlock_status status;
npunlock_view graph_blob = {blob_bytes, blob_byte_count};

options.struct_size = sizeof(options);
options.driver_index = GRAPHINFER_AUTO_INDEX;
options.device_index = GRAPHINFER_AUTO_INDEX;
options.timeout_ms = 15000;

input.struct_size = sizeof(input);
input.argument_index = GRAPHINFER_AUTO_INDEX;
input.argument_name_utf8 = (npunlock_view){(const uint8_t *)"input", 5};
input.data = (npunlock_view){tensor_bytes, tensor_byte_count};

status = graphinfer_infer(&options, graph_blob, &input, 1, &result);
if (status == NPUNLOCK_STATUS_OK) {
  size_t output_index;
  for (output_index = 0; output_index < result.output_count; ++output_index) {
    const graphinfer_output *output = &result.outputs[output_index];
    /* Consume output->data using output->dims and output->precision. */
  }
}
graphinfer_result_release(&result);
```

Successful driver submission is not proof of correct computation. Applications
must validate output semantics against their own oracle when correctness
matters.

## Concurrency and timeouts

Calls to `shavecc` and `ir2blob` launch independent worker processes through
one shared Win32 launcher. Request, protocol response,
stdout, and stderr use separate bounded pipes, and process trees are terminated
through a Job Object when a deadline expires. `patchblob` operates on caller
and result buffers without global mutable parser state. Graph inference is
in-process: the copied call owns a temporary session, while explicit shared
sessions serialize calls per session. Both use the supplied timeout for fence
waits. Each caller chooses copied or shared tensors and sets a finite timeout.

The initial public contract remains Windows x64, Meteor Lake/NPU3720, and
target `3720xx`. Custom ACT tensors are static dense FP16, plus the validated
unary FP32 accuracy-mode carrier. Other devices, dtypes, layouts, dynamic
shapes, and arbitrary source-node mapping are not implied.
