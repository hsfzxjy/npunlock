# Python API

[Documentation index](README.md)

The `npunlock` Python package is a small symbolic graph builder over the native
C libraries. It is not an eager tensor framework: constructing a `Tensor`
records graph structure and does not run NumPy or NPU computation.

## Inputs and constants

Create graph inputs with an explicit name, static shape, and IR dtype:

```python
import numpy as np
import npunlock as npu

x = npu.input("x", shape=(1, 1024), dtype="f16")
bias = npu.constant(np.zeros((1, 1024), dtype=np.float16), name="bias")
```

Shapes contain only positive integer dimensions. Constants are copied from
contiguous NumPy storage into the serialized weights buffer.

## Ordinary operators

Operator attributes are dynamic. An unknown public module attribute produces
an operator factory, so the package does not maintain a runtime operator
registry:

```python
y = npu.Abs(x, _name="absolute")
z = npu.SomeDriverSupportedOp(
    y,
    alpha=0.5,
    _shape=y.shape,
    _dtype=y.dtype,
)
```

This is equivalent to calling `npu.op("Abs", ...)` or
`npu.op("SomeDriverSupportedOp", ...)`. Use `npu.op()` when an operator name
collides with a real API member.

Arguments follow one convention:

```text
positional Tensor arguments     graph input edges
ordinary keyword arguments     serialized operator attributes
underscore keyword arguments   npunlock metadata
```

`npunlock` performs conservative output-spec inference for these operations:

- shape-preserving unary `Abs`, `Exp`, `Log`, `Relu`, `Sigmoid`, `Sqrt`, and
  `Tanh` inherit their one input's shape and dtype;
- elementwise `Add`, `Divide`, `Maximum`, `Minimum`, `Multiply`, and `Subtract`
  inherit a result only when their two symbolic input specs match exactly.
- `MatMul` infers `(M, N)` for two static rank-2 tensors with the same dtype,
  boolean transpose flags, and matching effective inner dimensions. Omitted
  transpose flags are serialized explicitly as `false`, so `a @ b` works for
  ordinary `(M, K) @ (K, N)` tensors.
- `Concat` accepts equal-rank, equal-dtype inputs, checks every non-axis
  dimension, and sums the selected axis;
- `Transpose` infers its shape when its order input is a one-dimensional
  integer constant containing a complete axis permutation;
- `Reshape` accepts a one-dimensional integer constant pattern, checks the
  element count, supports one `-1`, and implements explicit `special_zero`;
- `Squeeze` and `Unsqueeze` infer static shapes from one-dimensional integer
  constant axes, including normalized negative axes;
- `Split` accepts a scalar integer constant axis and positive `num_splits`,
  requires exact axis divisibility, and returns one contract per equal part;
- `VariadicSplit` accepts a scalar integer constant axis and one-dimensional
  constant lengths, including at most one `-1` remainder, and returns one
  contract per positive unequal part;
- `Pad` supports constant mode with nonnegative one-dimensional begin/end
  constants and a scalar constant fill value matching the data dtype;
- `Slice` accepts one-dimensional constant starts, stops, positive steps, and
  unique axes; it normalizes negative axes and Python-style negative bounds;
- `Gather` accepts integer indices and a scalar integer constant axis, infers
  the static output shape, and currently requires `batch_dims=0`; and
- `Convert` preserves shape and uses its normalized `destination_type` as the
  output dtype.

This is frontend convenience, not general graph or broadcasting inference.
Unknown operations, mismatched elementwise inputs, batched/broadcast MatMul,
and other shape-changing operations still require both `_shape` and `_dtype`.
Explicit metadata remains available as an override. Multiple outputs use
`_outputs`:

```python
a, b = npu.SomeOp(
    x,
    _outputs=[
        npu.TensorSpec((1, 32), "f16"),
        npu.TensorSpec((1, 64), "f16"),
    ],
)
```

The installed Intel compiler remains the authority on whether an ordinary
operator name, attributes, shapes, and opset are accepted.

`Tensor` implements thin symbolic sugar for matching tensor specs:

```python
c = a + b       # Add
c = a * b       # Multiply
c = a @ b       # MatMul
```

## Custom operations

Attach C source bytes or a source-file path with `npu.custom()`:

```python
y = npu.custom(
    x,
    source=b"""
#include <npunlock/npu3720_kernel.h>

void controlled_act(unsigned layerParams) {
    act_abi_invocation invocation;
    ACT_ABI_LOAD_INVOCATION32_OR_RETURN(layerParams, invocation);
    /* Use the INPUT_PTR32/OUTPUT_PTR32 helpers, then compute the result. */
}
""",
    carrier="Abs",
    _name="y",
)
```

Unlike ordinary dynamic operators, a single-output custom operation inherits
any omitted `_shape` or `_dtype` from its first input. The common same-shape,
same-dtype case therefore needs neither argument. Supply an explicit override
only for a carrier contract you have independently validated, or use `_outputs`
for multiple outputs. Metadata expressiveness does not establish hardware
support for a different shape, dtype, or output count.

The carrier is serialized into the graph sent to Intel. The C source is
compiled separately and installed after the native graph is returned. See
[Custom kernels](CUSTOM_KERNELS.md) for the bundled NPU3720 include, entry
function, tensor helpers, and math-link compatibility macro.

When deployment already has a linked kernel, attach it to the symbolic node:

```python
y = npu.custom(
    x,
    kernel=npu.CompiledKernel("kernels/custom.elf"),
    carrier="Abs",
    _name="y",
)
program = npu.compile(npu.Graph([x], [y]))
```

For delayed selection, leave the node source-free and pass an explicit
`KernelSpec` or `CompiledKernel` entry in `kernels`; otherwise `plan()` reports
the missing symbolic node before invoking MoviTools or patching the carrier.
Normal source-based `npu.compile()` usage is unchanged.
The one-call form is limited to graphs with unambiguous automatic ACT mapping.
Use `prepare()` and explicit bindings for any graph whose mapping needs review.

Normal code can rely on automatic selection only for graphs satisfying the
documented positional contract. The usual case maps one ACT group to each
computational node. If the compiler partitions one large custom operation into
several consecutive groups, `npunlock` also accepts them when one unique
exact-cover mapping can be proven from the declared output size and the
groups' compatible ABI metadata.

When compiler placement makes that mapping ambiguous, use a prepared graph as
described in [Inspect and bind a prepared graph](#inspect-and-bind-a-prepared-graph).
The lowest-level `_patch_targets` metadata remains available for callers with
independently validated invocation and range indices, but it is not needed for
the normal prepared-graph workflow.

One unequal-input case is implemented: a static same-precision two-input
carrier whose second input is a one-element scalar. Discovery records that
layout explicitly, and the kernel reads the broadcast value as `input_1[0]`
for every invocation-local output element. This has been executed with FP16
and FP32 `[1,32]` data plus a `[1,1]` scalar. It does not imply support for
other input positions, broadcast axes, shapes, or operators.

## Graphs

Create a graph from declared inputs and one or more output tensors:

```python
graph = npu.Graph(
    inputs=[x],
    outputs=[y],
    name="my_graph",
)
```

Every declared input must reach an output, and a graph cannot use an
undeclared `Parameter` tensor. Nodes are serialized in topological order.

## MoviTools configuration

Custom compilation requires the extracted `MVC_DEPEND` root:

```python
npu.configure(movi_dll_dir=r"C:\path\to\MVC_DEPEND")
```

Resolution order is:

1. `movi_dll_dir=` passed directly to `npu.compile()`;
2. the process-local value set by `npu.configure()`; then
3. `NPUNLOCK_MOVITOOLS_DIR`.

Pass `None` to `configure()` to clear the process-local override. The supplied
path must be the root containing `bin` and `lib`; passing `bin` itself is not
supported. See [Getting MoviTools](GET_MOVITOOLS.md).

## Environment diagnostics

Inspect the installed native runtime, configured MoviTools layout, Intel NPU,
graph extension, and graph compiler with:

```powershell
python -m npunlock doctor
```

The command compiles a tiny ordinary `Abs` carrier through the installed Intel
driver to query the real device/compiler path. It does not invoke MoviTools or
execute an NPU workload. All driver/compiler calls retain the finite timeout.

The final capability lines are intentionally separate:

- **run saved programs** requires the bundled native runtime plus the Intel NPU
  driver/compiler path;
- **compile custom C** additionally requires all expected DLLs and `mlibm.a`
  under the configured `MVC_DEPEND` root.

Use `--json` for schema `npunlock.doctor.v2`, or `--skip-driver` to inspect the
package and MoviTools files without querying an NPU:

```powershell
python -m npunlock doctor --json
python -m npunlock doctor --skip-driver
```

An explicit full check exercises the entire supported path: compile a small C
kernel with MoviTools, patch it into an FP16 carrier, execute it on the NPU,
and compare all 32 output elements bit-for-bit with a host add-one oracle:

```powershell
python -m npunlock doctor --full
```

This invokes proprietary compiler DLLs and NPU execution, so it is never part
of the default doctor run. `--full` and `--skip-driver` are mutually exclusive.

Source-tree developers may pass `--native-dir build\windows\Debug`.

## Compilation

Compile a symbolic graph with:

```python
program = npu.compile(graph)
```

The installed package contains `npunlock.dll` and `npunlock_worker.exe`, so
normal users do not set `native_dir`. A source-tree developer can override the
native directory explicitly or with `NPUNLOCK_NATIVE_DIR`.

Important optional arguments include:

| Argument | Purpose |
| --- | --- |
| `movi_dll_dir` | explicit `MVC_DEPEND` root |
| `timeout_ms` | deadline for worker-backed stages; default 20,000 ms |
| `build_flags` | Intel graph-compiler flags |
| `definitions` | uppercase `NAME=DECIMAL` C definitions |
| `linker_script` | advanced linker-script bytes or path |
| `native_dir` | development override containing the native DLL and worker |

For FP32 custom nodes, `compile()` automatically emits the validated
precision-preservation metadata and selects compiler accuracy mode when build
flags are otherwise empty. Caller-supplied flags for such a graph must include
`EXECUTION_MODE_HINT="ACCURACY"`.

Compilation returns a `Program` containing the final graph bytes, serialized
IR, driver/compiler provenance, and patch reports.

## Inspect and bind a prepared graph

`npu.prepare()` serializes the graph, asks the Intel graph compiler for the
native carrier once, and discovers its validated ACT groups without compiling
or patching the custom C yet:

```python
prepared = npu.prepare(graph)
print(prepared.explain())
```

`prepared.groups` contains immutable `ActGroup` values. Each group reports its
index, input count, FP16/FP32/I32/I16/I8 dtype when uniform, aggregate descriptor
element and byte coverage, stable per-invocation input/output element counts,
contract flags, and invocation/range indices. `group.element_count` and
`group.span_bytes` are not graph-level tensor sizes: tile replicas may duplicate
the same logical work. The per-invocation count properties return `None` when
targets vary, in which case inspect `group.targets` individually.
`prepared.mappings` records whether each custom node has an automatic mapping
or needs an explicit caller choice.

Filter groups by validated contract fields and bind them to the actual
symbolic custom outputs:

```python
unary_f32 = prepared.find_group(input_count=1, dtype="f32")
binary_f16 = prepared.find_group(input_count=2, dtype="f16")

program = prepared.build(
    bindings={
        scaled_output: unary_f32,
        mixed_output: binary_f16,
    }
)
```

A binding may use one `ActGroup`, its integer index, or an ordered sequence of
groups for a compiler-partitioned custom operation. Plan creation checks input
arity, tensor precision, exact-cover requirements for multi-group operations,
duplicate group assignment, and completeness before compiling the C kernels.

Discovered positional mapping is limited to graph compiler 8.3. A different
compiler version is rejected before discovery because the native identity bytes
and grouping rules have not been validated there. Raw `_patch_targets` bypass
positional discovery for callers who have independently validated another
layout; they are deliberately an advanced, caller-owned contract.

The binding key is a symbolic `Tensor` from the caller's graph, not an IR node
name recovered from the native blob. Native blobs do not expose a proven
source-name mapping. Group filtering is a convenience over validated metadata,
not proof that a group implements the caller's intended operation; ambiguous
graphs still require an informed explicit choice.

`find_group()` requires exactly one match and reports the applied filter plus
candidate contract summaries when the result is missing or ambiguous. Use
`find_groups()` when zero or several results are expected and the caller will
handle them deliberately.

### Review a multi-kernel build plan

For a graph with several custom kernels, separate validation from compiler
execution with `PreparedGraph.plan()`. `KernelSpec` snapshots source and linker
bytes immediately and gives one kernel its own definitions or linker override:

```python
plan = prepared.plan(
    bindings={
        scaled_output: unary_f32,
        mixed_output: binary_f16,
    },
    kernels={
        scaled_output: npu.KernelSpec(
            "kernels/scale.c",
            definitions=("SCALE=3",),
        ),
    },
)

print(plan.explain())
program = plan.build()
```

`plan()` performs complete binding validation without requiring MoviTools. Its
deterministic explanation includes each symbolic custom node, selected groups
and invocation/range indices, tensor dtypes and spans, source SHA-256,
definitions, and linker-script SHA-256. This makes the intended mutation
reviewable before any proprietary compiler runs.

A kernel omitted from `kernels` uses the source embedded by `npu.custom()` and
the plan's default `definitions` and `linker_script`. A supplied `KernelSpec`
is a complete per-kernel override. During `BuildPlan.build()`, identical source,
definitions, and linker bytes share one compiled ELF; patching still occurs
separately for every selected custom node and retains one report per patch.

Use `CompiledKernel` when a linked SHAVE ELF was produced earlier and should be
reused without invoking MoviTools:

```python
plan = prepared.plan(
    bindings={custom_output: selected_group},
    kernels={custom_output: npu.CompiledKernel("kernels/custom.elf")},
)
print(plan.explain())
program = plan.build()
```

`CompiledKernel` snapshots the ELF bytes when it is constructed, just as
`KernelSpec` snapshots source and linker bytes. If every custom node has a
compiled-kernel override, `build()` does not require a configured
`MVC_DEPEND` root. A mixed plan accepts both types and invokes MoviTools only
for its `KernelSpec` entries. The native patcher still validates the supplied
ELF against the established NPU3720 contract; precompilation does not bypass
ELF, target, relocation, or graph-target checks. `plan.explain()` identifies
compiled inputs by path and SHA-256.

The high-level API can produce and persist that reusable value directly:

```python
compiled = npu.compile_kernel(
    "kernels/custom.c",
    definitions=("MODE=2",),
)
compiled.save("kernels/custom.elf")

# Attach the immutable ELF directly for the common unambiguous case.
custom_output = npu.custom(
    x, kernel=compiled, carrier="Abs", _name="custom_output"
)
program = npu.compile(npu.Graph([x], [custom_output]))
```

`compile_kernel()` uses the same native `shavecc` implementation, bounded
worker, embedded default linker script, and MoviTools directory precedence as
graph compilation. It does not compile an IR graph or require NPU execution.
It accepts source bytes, a source path, or an existing `KernelSpec`. Settings
passed separately cannot override an existing spec because its bytes and
compiler settings were already snapshotted.
The returned `CompiledKernel` may be reused across compatible prepared
carriers, but each plan still validates the ELF and its selected ACT targets.
`kernel=` also accepts a `KernelSpec` when per-node definitions or a linker
override should travel with the symbolic node. `source=` and `kernel=` are
mutually exclusive. Keep the source-free node plus `kernels={output: ...}`
form when selecting or replacing the kernel only at plan time.

`PreparedGraph.build(kernels=...)` is the compact form for an automatically
mapped graph. Use `plan(kernels=...)` when the selection should be printed and
reviewed before building, and add explicit `bindings=` whenever placement is
ambiguous. Both forms execute the same plan validation and patch path.

For an unambiguous graph that does not need separate carrier reuse or plan
inspection, `npu.compile(graph, kernels={output: compiled})` combines
`prepare()` and `PreparedGraph.build()` in one call. A plan containing only
`CompiledKernel` values needs no MoviTools configuration; mixed mappings invoke
MoviTools only for their `KernelSpec` entries.

`PreparedGraph.build()` remains the compact API and delegates through the same
plan validation. Both paths patch the carrier already held by `PreparedGraph`;
neither repeats graph compilation.

Mixed-precision unary conversion groups use version 2 per-tensor targets.
Their `ActGroup.input_dtypes` and `ActGroup.output_dtype` properties expose the
separate contracts, and `ActGroup.unary_conversion` identifies the currently
supported FP16/FP32 conversion form:

```python
to_f16 = prepared.find_group(
    input_dtypes=("f32",), output_dtype="f16"
)
program = prepared.build(bindings={converted: to_f16})
```

These groups always require explicit binding. The compiler-generated targets
are full-tensor tile replicas, so the frontend patches the complete selected
group without summing its element counts. See
[`example_conversion_kernels.py`](../examples/example_conversion_kernels.py)
for both directions in one connected graph.

When version 2 discovery also reports an ordinary same-precision group, plan
creation uses the original version 1 patch contract if every per-tensor record
is exactly representable by it: static dense CMX FP16/FP32, equal input/output
counts and spans, and optionally the established scalar second input. This is
a lossless contract projection, not a new ABI assumption; `patchblob` performs
the normal version 1 validation again. Mixed-precision conversion groups remain
version 2. The plan explanation shows which version will actually be patched.

The same-precision unary I32, I16, and I8 carriers and the equal-span binary
I16 carrier always remain version 2 and can use automatic positional mapping
when each is the unique compatible group. See
[`example_i32_custom.py`](../examples/example_i32_custom.py) and
[`example_i8_custom.py`](../examples/example_i8_custom.py), plus
[`example_i16_custom.py`](../examples/example_i16_custom.py) and
[`example_i16_binary_custom.py`](../examples/example_i16_binary_custom.py).

For a single set of kernel-build settings, pass them to `prepared.build()` in
the same way as `npu.compile()`:

```python
program = prepared.build(
    bindings={custom_output: selected_group},
    movi_dll_dir=r"C:\path\to\MVC_DEPEND",
    definitions=("MODE=1",),
)
```

See
[`example_mixed_precision_multi_custom.py`](../examples/example_mixed_precision_multi_custom.py)
for a complete graph whose independent FP32 and FP16 branches require explicit
binding because the compiler's ACT order differs from symbolic output order.
For a connected v2 conversion → v1 binary → DPU `MatMul` → v2 conversion path,
including prepared-carrier and final-program reload, see
[`example_connected_mixed_precision.py`](../examples/example_connected_mixed_precision.py).

## Reuse a prepared carrier

Export the expensive driver-compiled carrier separately when kernel C will be
edited and rebuilt repeatedly:

```python
prepared.export("network.prepared.npunlock", plan=plan)

# A later process reconstructs the same symbolic graph. Kernel source may have
# changed because source bytes are deliberately not part of carrier identity.
prepared = npu.load_prepared("network.prepared.npunlock", graph=graph)
print(prepared.explain())
program = prepared.plan().build()
```

The versioned bundle contains the unpatched native carrier, exact serialized
IR XML and weights identities, effective build flags, original driver/compiler
provenance, discovered ACT contracts, and optionally the reviewed symbolic
bindings from a `BuildPlan`. It does not contain kernel source, compiled kernel
ELFs, MoviTools, or Intel driver binaries. Per-kernel source, compiled ELF,
definitions, and linker overrides are caller inputs; pass `KernelSpec` or
`CompiledKernel` values to the loaded graph's new plan.

`load_prepared()` reserializes the caller graph and requires exact XML and
weights hashes. It uses a bounded read-only driver query—without creating or
compiling a graph—and requires the current driver, device, graph-extension,
and compiler versions to match the producer provenance exactly. It then
rediscovers the ACT groups from the bundled carrier, compares every target
contract with the manifest, and revalidates saved bindings for arity, dtype,
exact cover, uniqueness, and range. A stale driver, changed graph, corrupt
carrier, altered contract, or incompatible saved binding is rejected before
MoviTools runs. There is deliberately no stale-provenance override in the v1
format; executing the eventual program remains the separate semantic check.

## Export and load a program bundle

Use a self-describing `.npunlock` bundle when another process or machine needs
to execute a compiled program without reconstructing its symbolic `Graph`:

```python
program.export("my_program.npunlock")

loaded = npu.load("my_program.npunlock")
output = loaded.run_one(input_value)
```

The versioned bundle contains the patched native graph, static input/output
names, shapes and dtypes, SHA-256 hashes, graph compiler and device provenance
when available, and every patch report. It does not contain custom C source,
MoviTools, Intel driver DLLs, or other proprietary binaries. Loading therefore
does not require MoviTools or repeat graph/kernel compilation, but execution
still requires a compatible Intel NPU driver and hardware.

`loaded.input_contracts` and `loaded.output_contracts` expose immutable
`TensorContract` values. `loaded.graph` is `None`: the bundle deliberately does
not invent a symbolic graph or claim that one can be reconstructed from the
native binary. The original manifest bytes remain available as
`loaded.artifact_manifest` for provenance inspection.

Bundle loading checks the schema, declared member set, size limits, tensor
contract, and SHA-256 of the graph and patch reports before constructing a
`Program`.

## Save and load raw native graphs

The final patched native graph can be retrieved as immutable bytes or written
directly to a file:

```python
blob = program.to_bytes()
program.save("my_graph.blob")
```

Load either representation without repeating IR compilation, custom C
compilation, or patching:

```python
from_memory = npu.load_native(blob, graph=graph)
from_file = npu.load_native_file("my_graph.blob", graph=graph)

output = from_file.run({"x": input_value})
```

Unlike a `.npunlock` bundle, a raw blob has no persisted tensor contract. The
symbolic `graph` is therefore required because `Program.run()` needs its input
and output names, shapes, and dtypes. `npunlock` does not try to reconstruct
that high-level graph from the native binary, and it cannot prove that a
caller-supplied graph describes an unrelated blob. Always load a raw blob with
the same graph contract that produced it.

A loaded program has `serialized_ir` and `ir_provenance` set to `None` and an
empty `patch_reports` tuple because those build-time records cannot be
recovered from the native blob. Loading still uses the installed Intel NPU
driver when `run()` executes, but it does not require MoviTools.

## Native tool output and errors

Graph compilation and custom C compilation run in bounded native worker
processes. Their stdout and stderr streams are captured separately. Graph
execution is always in-process; callers choose either ordinary NumPy arrays,
which are copied through internal Level Zero host allocations, or explicit shared
arrays. See [Shared input and output buffers](#shared-input-and-output-buffers).

If a worker-backed compilation operation fails, Python writes captured stdout to the
process stdout and captured stderr to the process stderr **verbatim**, without
embedding or quoting either stream inside the exception message. It then
raises `npu.NativeError`, whose `stdout_log` and `stderr_log` attributes retain
the original bytes alongside the structured `diagnostic` bytes.

If the native operation succeeds but its worker wrote to stderr, Python emits
a `RuntimeWarning`. Successful stdout remains captured internally and does not
add noise to normal program output. A vendor tool's structured diagnostic
buffer is distinct from its operating-system stderr stream.

## Execution

For an exactly one-input, one-output graph, `Program.run_one()` accepts the
input array and returns the output array directly:

```python
input_value = np.linspace(-4, 4, 1024, dtype=np.float16).reshape(1, 1024)
result = program.run_one(input_value)
```

It rejects programs with multiple inputs or outputs instead of choosing one.
For general graphs, `Program.run()` accepts a mapping from every declared
input name and returns a mapping by output name:

```python
outputs = program.run({"x": x_value, "peer": peer_value})
result = outputs["y"]
```

Input names must match exactly. Shape and dtype must match the symbolic graph;
noncontiguous arrays are copied to contiguous storage. The current execution
boundary supports static FP16, FP32, U8, I8, U16, I16, U32, and I32 graph inputs
and outputs with at most five dimensions. U8, U16, and U32 remain ordinary
graph-execution features. I32, I16, and I8 additionally have one documented
same-precision unary custom-kernel carrier each; this does not imply broader
integer custom support. I16 additionally supports one equal-span two-input
carrier; scalar, broadcast, and mixed-integer inputs remain unsupported.

The supported dtype list describes tensor transport, not every operator/dtype
combination. In particular, U8 data with I32 indices has passed `Gather`, while
the tested U16-data `Gather` returned incorrect values despite successful graph
submission. Treat the installed compiler and a host oracle as authoritative.
U32 is validated through structural `Split`; no U32 arithmetic operation is
implied by that result.
I64 is validated only for caller-provided `Gather` indices. A full-width I64
Split silently narrowed values to signed 32 bits, so ordinary I64 data and
outputs are not supported.

For a multi-output node, output mappings use the compiler-compatible source
names, such as `split.0`, `split.1`, and `split.2`. Give the node `_name=` when
those keys should remain stable across graph edits. This also lets shared
execution retain exact-name argument selection for every output.

Returned arrays own their data. Native buffers are copied before the matching
C release function is called. The first call lazily creates an execution
session; later calls reuse the initialized graph and its internal tensor
allocations. Calls through one `Program` are serialized by the native session.

Use a context manager when the program has a clear execution scope, or call
`close()` explicitly:

```python
with npu.load("my_program.npunlock") as program:
    for batch in batches:
        outputs = program.run({"x": batch})

# Equivalent when a context manager is inconvenient.
program.close()
```

Closing is idempotent. A closed program cannot run or allocate more shared
arrays.

### Shared input and output buffers

For a one-input/one-output program, allocate correctly shaped host/NPU shared
arrays without repeating the tensor contract:

```python
shared_x = program.shared_input()
shared_y = program.shared_output()
np.copyto(shared_x, input_value)
result = program.run_one(shared_x, output=shared_y)
assert result is shared_y
```

For a general graph, the dictionary helpers use every saved graph argument
name, shape, and dtype:

```python
shared_inputs = program.shared_inputs()
shared_outputs = program.shared_outputs()
shared_x = shared_inputs["x"]
shared_y = shared_outputs["y"]

np.copyto(shared_x, input_value)
np.clip(shared_x, -4, 4, out=shared_x)
outputs = program.run(shared_inputs, outputs=shared_outputs)
assert outputs["y"] is shared_y
```

`program.shared_array(shape, dtype)` remains available for callers that need a
manually shaped shared allocation.

`SharedArray` is an `np.ndarray` subclass backed directly by a host-visible
Level Zero shared allocation. Indexing, broadcasting assignments, ufuncs, and
functions accepting NumPy arrays work normally. Use an `out=` argument when a
NumPy operation should write back into NPU-visible storage; a function that
creates a new result may return ordinary host storage.

Every shared input and output must be a complete contiguous array allocated by
the same `Program`, with exactly the graph tensor's shape and dtype. Shared
execution binds those allocations directly and returns the caller's output
objects without copying tensor bytes. Array views retain the native allocation
owner, so the allocation is released only after the final view is gone. Their
memory remains valid after `program.close()`, although the closed graph cannot
execute again.

Both tensor modes keep Level Zero graph execution in the Python process. Fence
waits use `timeout_ms`, but a driver call that never returns cannot be
terminated without ending the application. Use ordinary `Program.run()` for
simple owned NumPy results; choose shared arrays when avoiding tensor copies or
reusing allocations matters.

## Serialization without execution

The graph serializer is available independently:

```python
serialized = npu.serialize_ir(graph)
xml = serialized.xml
weights = serialized.weights
```

It emits OpenVINO-format IR XML and weights but does not import or invoke
OpenVINO. Python never parses or modifies the native graph blob itself.

## Examples

- [FP16 GELU](../examples/example_gelu.py)
- [FP32 GELU](../examples/example_gelu_f32.py)
- [FP16 custom kernel with a descriptor-level row broadcast](../examples/example_broadcast_custom.py)
- [FP16 five-input custom kernel](../examples/example_five_input_custom.py)
- [I32 structural graph](../examples/example_i32_structural.py)
- [U8 Reshape and conversion graph](../examples/example_u8_reshape.py)
- [I8 Squeeze/Unsqueeze and conversion graph](../examples/example_i8_axes.py)
- [I16 multi-output Split graph](../examples/example_i16_split.py)
- [I16 unary custom kernel](../examples/example_i16_custom.py)
- [I16 two-input custom kernel](../examples/example_i16_binary_custom.py)
- [U16 multi-output Split graph](../examples/example_u16_split.py)
- [mixed-input U8/I32 Gather graph](../examples/example_u8_gather.py)
- [U32 multi-output Split graph](../examples/example_u32_split.py)
- [U8 Gather with I64 indices](../examples/example_u8_gather_i64.py)
- [U8 unequal VariadicSplit graph](../examples/example_u8_variadic_split.py)
- [U8 multi-axis Slice graph](../examples/example_u8_slice.py)
- [U8 constant-mode Pad graph](../examples/example_u8_pad.py)
- [Multi-layer, two-input custom kernel](../examples/example_multilayer_multi_input.py)
- [Mixed-precision custom branches with prepared binding](../examples/example_mixed_precision_multi_custom.py)
- [Branched DPU/ACT network with three custom kernels](../examples/example_branched_network.py)

See [Current limitations](LIMITATIONS.md) before changing carrier, precision,
shape, or graph structure.

[Back to documentation index](README.md)
