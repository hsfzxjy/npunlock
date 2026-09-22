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
y = npu.Abs(x, _shape=x.shape, _dtype=x.dtype, _name="absolute")
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

For a single output, `_shape` and `_dtype` are required because `npunlock`
does not implement general operator shape inference. Multiple outputs use
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
    npunlock_npu3720_act_abi_invocation invocation;
    NPUNLOCK_NPU3720_ACT_ABI_LOAD_INVOCATION32_OR_RETURN(
        layerParams, 1024u, invocation);
    /* Use the INPUT_PTR32/OUTPUT_PTR32 helpers, then compute the result. */
}
""",
    carrier="Abs",
    _shape=x.shape,
    _dtype=x.dtype,
    _name="y",
)
```

The carrier is serialized into the graph sent to Intel. The C source is
compiled separately and installed after the native graph is returned. See
[Custom kernels](CUSTOM_KERNELS.md) for the bundled NPU3720 include, entry
function, tensor helpers, and math-link compatibility macro.

Advanced callers may supply `_patch_targets`, but invocation and range indices
are native compiler-output details. Normal code should rely on automatic
selection only for graphs satisfying the documented one-to-one positional
contract.

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

## Worker output and errors

Graph compilation, custom C compilation, and execution run in bounded native
worker processes. Their stdout and stderr streams are captured separately.

If a worker-backed operation fails, Python writes captured stdout to the
process stdout and captured stderr to the process stderr **verbatim**, without
embedding or quoting either stream inside the exception message. It then
raises `npu.NativeError`, whose `stdout_log` and `stderr_log` attributes retain
the original bytes alongside the structured `diagnostic` bytes.

If the native operation succeeds but its worker wrote to stderr, Python emits
a `RuntimeWarning`. Successful stdout remains captured internally and does not
add noise to normal program output. A vendor tool's structured diagnostic
buffer is distinct from its operating-system stderr stream.

## Execution

`Program.run()` accepts a mapping from every declared input name to a NumPy
array:

```python
input_value = np.linspace(-4, 4, 1024, dtype=np.float16).reshape(1, 1024)
outputs = program.run({"x": input_value})
result = outputs["y"]
```

Input names must match exactly. Shape and dtype must match the symbolic graph;
noncontiguous arrays are copied to contiguous storage. The current execution
boundary supports static FP16 and FP32 graph inputs and outputs with at most
five dimensions.

Returned arrays own their data. Native buffers are copied before the matching
C release function is called.

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
- [Multi-layer, two-input custom kernel](../examples/example_multilayer_multi_input.py)

See [Current limitations](LIMITATIONS.md) before changing carrier, precision,
shape, or graph structure.

[Back to documentation index](README.md)
