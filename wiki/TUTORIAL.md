# Tutorial: write and run a custom NPU kernel

[Documentation index](README.md)

This tutorial uses the Python API to take a small C function from source to
execution on an Intel Core Ultra NPU. It starts with the simplest useful case,
then adds inputs, carrier inspection, mixed precision, reusable artifacts, and
shared buffers.

The current supported platform is Windows x64 with Meteor Lake / NPU3720 and
graph compiler 8.3. Shapes are static. The examples below are recipes for the
execution-tested carriers, not promises about every operation, shape, driver,
or Intel NPU generation.

## 1. Set up the two toolchains

`npunlock` uses two separate Intel components:

- the installed Intel NPU driver compiles the surrounding graph and executes
  the final program; and
- MoviTools compiles your C source into ACT-SHAVE machine code.

Install `npunlock` from the checkout:

```powershell
python -m pip install .
```

Obtain and extract the MoviTools `MVC_DEPEND` directory as described in
[Getting MoviTools](GET_MOVITOOLS.md). The old Lenovo package is used only as
the source of those compiler files—do not install its old driver. Keep the
current Intel NPU driver installed for graph compilation and execution.

Point `npunlock` at the extracted root, not its `bin` subdirectory:

```powershell
$env:NPUNLOCK_MOVITOOLS_DIR = 'C:\path\to\MVC_DEPEND'
python -m npunlock doctor --full
```

You can configure the same path in Python instead:

```python
import npunlock as npu

npu.configure(movi_dll_dir=r"C:\path\to\MVC_DEPEND")
```

The wheel or source install supplies `npunlock.dll`, the compiler worker, and
the NPU3720 kernel header. MoviTools and the Intel driver remain external.

## 2. Understand the carrier

The Intel graph compiler does not accept an arbitrary C node. `npunlock`
therefore gives it a supported graph operation called a **carrier**, then
replaces that operation's ACT-SHAVE code with your compiled C:

```text
symbolic graph with carrier
          |
          v
Intel graph compiler creates scheduling + tensor descriptors
          |
          v
npunlock substitutes your ACT-SHAVE machine code
          |
          v
Intel driver executes the patched graph
```

The carrier is a resource and ABI template. It determines input count, tensor
dtypes, shapes visible to the graph, invocation chunks, memory placement, and
scheduling. It does **not** need to resemble your custom calculation. An `Abs`
carrier can run GELU or an affine transform because its ACT code is replaced.

This distinction is the central rule for choosing a carrier: match its tensor
contract, not its mathematical name.

## 3. Write the first kernel

Start with a unary FP16 kernel. This one computes `2*x + 1`:

```c
#include <npunlock/npu3720_kernel.h>

void controlled_act(unsigned layerParams) {
    act_abi_invocation invocation;
    ACT_ABI_LOAD_INVOCATION32_OR_RETURN(layerParams, invocation);

    const __fp16 *input =
        ACT_ABI_INPUT_PTR32(const __fp16, invocation, 0u);
    __fp16 *output =
        ACT_ABI_OUTPUT_PTR32(__fp16, invocation, 1u);

    for (unsigned i = 0; i < invocation.element_count; ++i) {
        output[i] = (__fp16)((float)input[i] * 2.0f + 1.0f);
    }
}
```

There are four important pieces:

1. Include `npunlock/npu3720_kernel.h`. It exposes the assumptions specific to
   the observed NPU3720 ABI.
2. Export `controlled_act(unsigned layerParams)`, the default kernel entry.
3. Load and validate the current invocation before touching memory.
4. Use tensor record 0 for the unary input and record 1 for the output.

`invocation.element_count` is the size of the current compiler-created chunk,
not necessarily the total number of graph elements. Every invocation runs the
same entry over its own pointers and count.

## 4. Put the kernel in a graph

Embed the source as bytes, create a symbolic input, and use `Abs` as the unary
FP16 carrier:

```python
import numpy as np
import npunlock as npu

kernel_c: bytes = b"""
#include <npunlock/npu3720_kernel.h>

void controlled_act(unsigned layerParams) {
    act_abi_invocation invocation;
    ACT_ABI_LOAD_INVOCATION32_OR_RETURN(layerParams, invocation);
    const __fp16 *input = ACT_ABI_INPUT_PTR32(const __fp16, invocation, 0u);
    __fp16 *output = ACT_ABI_OUTPUT_PTR32(__fp16, invocation, 1u);
    for (unsigned i = 0; i < invocation.element_count; ++i) {
        output[i] = (__fp16)((float)input[i] * 2.0f + 1.0f);
    }
}
"""

x = npu.input("x", shape=(1, 32), dtype="f16")
y = npu.custom(x, source=kernel_c, carrier="Abs", _name="y")
graph = npu.Graph([x], [y], name="first_custom_kernel")
program = npu.compile(graph)

input_value = np.linspace(-4, 4, 32, dtype=np.float16).reshape(1, 32)
output = program.run_one(input_value)
expected = (input_value.astype(np.float32) * 2.0 + 1.0).astype(np.float16)
np.testing.assert_array_equal(output, expected)
```

`npu.custom()` inherits the output shape and dtype from its first input for
this ordinary same-shape, same-dtype case. `npu.compile()` then performs the
whole pipeline: graph compilation, C compilation, validated ACT selection,
patching, and construction of an executable `Program`.

Always compare the complete output with an independent host oracle. Successful
compilation, graph loading, or fence completion proves transport—not the
meaning of your C code.

For nonlinear math, continue with the runnable [FP16 GELU
example](../examples/example_gelu.py) or [FP32 GELU
example](../examples/example_gelu_f32.py).

## 5. Program the invocation you actually receive

For the validated layouts, tensor records are consecutive:

```text
unary:       input 0, output
binary:      input 0, input 1, output
five-input:  inputs 0..4, output
```

That is why `ACT_ABI_OUTPUT_PTR32(type, invocation, input_count)` receives the
number of inputs. The helper locates the following output record.

Use these C types for the currently validated custom paths:

| Graph dtype | Kernel pointer type | Current custom scope |
| --- | --- | --- |
| `f16` | `__fp16` | unary, selected binary/broadcast, and equal-span or multi-scalar five-input carriers |
| `f32` | `float` | unary, scalar/equal-span binary, and exact row/per-row broadcasts |
| `i32` | `int` | unary and equal-span binary |
| `i16` | `short` | unary and equal-span binary |
| `i8` | `signed char` | unary and equal-span binary |
| `u16` | `unsigned short` | equal-span, either-position scalar, and exact row/per-row broadcasts |
| `u32` | `unsigned int` | equal-span, either-position scalar, and exact row/per-row broadcasts |
| `u8` | `unsigned char` input | only the validated U8-to-FP32 conversion |

Stay within the arithmetic range of integer outputs unless you are explicitly
testing overflow behavior. No saturation or quantization semantics are
implied.

An invocation is local. Do not assume it starts at graph element zero, covers
the whole graph tensor, or can read a neighbor owned by another invocation.
Reductions, sorting, windows, and scratch storage can be useful, but their
semantics stop at the current invocation boundary. See [Writing custom
kernels](CUSTOM_KERNELS.md#invocation-local-memory) for the measured limits and
advanced examples.

## 6. Choose a carrier by contract

Use this decision order:

1. Match the number of tensor inputs.
2. Match every input dtype and the output dtype.
3. Match the tested shape relationship: equal-span, one scalar input in the
   documented position, or the one documented row-broadcast layout.
4. Prefer a carrier already exercised by a runnable example.
5. Inspect the compiled ACT groups before binding a complex or conversion
   graph.

The current carrier recipes are:

| Desired custom contract | Carrier recipe | Mapping |
| --- | --- | --- |
| unary FP16 or FP32 | `Abs` | automatic when the graph is unambiguous |
| unary I32, I16, or I8 | `Abs` | automatic when the graph is unambiguous |
| two equal-span FP16 inputs | `Maximum` | automatic when the graph is unambiguous |
| two equal-span FP32 inputs | `Subtract` | automatic when the graph is unambiguous |
| FP16 data plus a one-element second input | `Maximum` | automatic when the exact scalar contract is discovered |
| FP16, FP32, U16, or U32 scalar first plus `(1,32)` data | `Subtract` | version 2; broadcast output shape inferred automatically |
| FP16, FP32, U16, or U32 `(4,8)` data plus `(4,1)` per-row scalars, either position | `Subtract` | automatic for this exact shape; one scalar per invocation |
| FP16 `(4,8)` plus `(1,8)` row input | `Maximum` | only the documented dense `(8,8)->8` invocation layout |
| FP32 `(4,8)` plus `(1,8)` row input | `Subtract` | only the documented dense `(8,8)->8` invocation layout |
| two equal-span I32 inputs | `Minimum` | automatic when the graph is unambiguous |
| two equal-span I16 inputs | `Maximum` | automatic when the graph is unambiguous |
| two equal-span I8 inputs | `Subtract` | automatic when the graph is unambiguous |
| two equal-span U16 inputs | `Subtract` | automatic when the graph is unambiguous |
| two equal-span U32 inputs | `Subtract` | automatic when the graph is unambiguous |
| U16 or U32 data plus a one-element second input | `Subtract` | automatic when the exact scalar contract is discovered |
| U16 or U32 `(4,8)` plus `(1,8)` row input | `Subtract` | only the documented dense `(8,8)->8` invocation layout |
| five equal-span FP16 inputs | `FakeQuantize`, `levels=256` | automatic for the documented carrier |
| FP16 `(1,32)` data plus four `(1,1)` scalar parameters | `FakeQuantize`, `levels=16` | automatic only for the documented `(16,1,1,1,1)->16` layout |
| FP16 `(4,8)` data plus four `(4,1)` per-row parameters | `FakeQuantize`, `levels=16` | automatic only for the documented four `(8,1,1,1,1)->8` invocations |
| two FP16 `(1,32)` tensors plus three `(1,1)` graph-global parameters | `FakeQuantize`, `levels=16` | automatic only for the documented `(16,16,1,1,1)->16` layout |
| two FP16 `(4,8)` tensors plus three `(4,1)` per-row parameters | `FakeQuantize`, `levels=16` | automatic only for the documented four-group `(4,4,1,1,1)->4` layout |
| FP32→FP16, FP16→FP32, or U8→FP32 | `Convert`, matching `destination_type` | explicit prepared-group binding |

These rows are verified recipes, not a general operator compatibility matrix.
Changing dtype, shape, compiler version, broadcast form, or graph context can
change or remove the ACT group. For example, there is no same-precision U8
custom carrier and no unary U16 or U32 carrier.

Do not choose an operation merely because its name resembles your function.
Also do not assume a different operation is usable because the Intel compiler
accepts it: it may lower to the DPU, fuse into a neighbor, disappear, or expose
a different ACT layout.

## 7. Add a second input

For an equal-span FP16 binary carrier, read input records 0 and 1 and write
record 2:

```c
#include <npunlock/npu3720_kernel.h>

void controlled_act(unsigned layerParams) {
    act_abi_invocation invocation;
    ACT_ABI_LOAD_INVOCATION32_OR_RETURN(layerParams, invocation);
    const __fp16 *a = ACT_ABI_INPUT_PTR32(const __fp16, invocation, 0u);
    const __fp16 *b = ACT_ABI_INPUT_PTR32(const __fp16, invocation, 1u);
    __fp16 *output = ACT_ABI_OUTPUT_PTR32(__fp16, invocation, 2u);

    for (unsigned i = 0; i < invocation.element_count; ++i) {
        output[i] = (__fp16)((float)a[i] * 0.75f + (float)b[i] * 0.25f);
    }
}
```

The graph side uses `Maximum` only to request the validated two-input ACT
environment:

```python
shape = (1, 32)
a = npu.input("a", shape=shape, dtype="f16")
b = npu.input("b", shape=shape, dtype="f16")
y = npu.custom(a, b, source=kernel_c, carrier="Maximum", _name="y")
program = npu.compile(npu.Graph([a, b], [y], name="weighted_mix"))
```

The FP32 equal-span form uses `float *`, `dtype="f32"`, and the tested
`Subtract` carrier instead. See the complete
[binary FP32 example](../examples/example_f32_binary_custom.py); the frontend
adds precision-preservation metadata and accuracy mode automatically.

The exact U16 recipe also uses `Subtract`, with `unsigned short *` pointers
and `dtype="u16"`. Keep intermediate arithmetic in a wider unsigned type and
stay within the output range unless overflow is the behavior being tested.
See the [binary U16 example](../examples/example_u16_binary_custom.py).

The U32 recipe uses the same `Subtract` carrier with `unsigned int *` pointers
and `dtype="u32"`. Keep calculations within the U32 range unless overflow is
the behavior under test. See the
[binary U32 example](../examples/example_u32_binary_custom.py).

For U16 and U32, the tested `Subtract` carrier also supports a `[1,1]` second
input read as `input_1[0]`, and the exact `(4,8)` plus `(1,8)` row form read as
`input_1[i]`. The [unsigned broadcast
example](../examples/example_unsigned_broadcast_custom.py) chains both forms
and runs them for both widths.

A scalar may instead be input zero for the exact `[1,1]` minus `[1,32]`
`Subtract` carrier in FP16, FP32, U16, or U32. `custom()` infers the broadcast
output shape from these two compatible binary inputs. More importantly, load
the invocation extent from output record 2:

```c
act_abi_invocation invocation;
ACT_ABI_LOAD_OUTPUT_INVOCATION32_OR_RETURN(layerParams, 2u, invocation);
```

The ordinary load helper reads dimensions from record zero, which is the
one-element scalar here and would process only one output. See the
[scalar-first example](../examples/example_scalar_first_custom.py) for all four
validated precisions.

The exact `[4,8]` data with `[4,1]` input follows the same pointer rule but
supplies a different scalar to each row invocation. Both operand orders are
validated for FP16, FP32, U16, and U32. Read the scalar record only at index
zero; use the output-record loader when that scalar is input zero. See the
[per-row scalar example](../examples/example_per_row_scalar_custom.py).

One higher-arity recipe combines FP16 data with four independently bound
parameters. Use `FakeQuantize(levels=16)` as the five-input carrier. The
graph-global form uses `(1,32)` data and four `(1,1)` inputs; record zero and
output record five contain 16 elements per invocation. The per-row form uses
`(4,8)` data and four `(4,1)` inputs; each of four invocations receives one
row's parameters and exposes eight data/output elements. In both forms,
records 1 through 4 contain one element each and must be read only at index
zero:

```c
const __fp16 *data = ACT_ABI_INPUT_PTR32(const __fp16, invocation, 0u);
const __fp16 *parameter_3 = ACT_ABI_INPUT_PTR32(const __fp16, invocation, 3u);
__fp16 *output = ACT_ABI_OUTPUT_PTR32(__fp16, invocation, 5u);
```

The [multi-scalar example](../examples/example_multi_scalar_custom.py) uses
the four parameters as scale, bias, lower, and upper bounds. This exact FP16
graph-global contract is distinct from the equal-span five-input carrier. The
[per-row multi-scalar example](../examples/example_per_row_multi_scalar_custom.py)
uses the same computation with four independently bound parameters per row.
Neither form establishes arbitrary parameter counts, positions, shapes, axes,
or precisions.

To combine two tensors with three graph-global parameters, use the exact
`(1,32)`, `(1,32)`, `(1,1)`, `(1,1)`, `(1,1)` source shapes. The compiler
presents records 0 and 1 as 16-element vectors and records 2 through 4 as
one-element parameters in each invocation. The
[graph-global hybrid example](../examples/example_global_hybrid_five_input_custom.py)
implements residual addition, affine scaling, and an upper clamp.

The per-row alternative uses the exact `(4,8)`, `(4,8)`, `(4,1)`, `(4,1)`,
`(4,1)` source shapes. The compiler
presents records 0 and 1 as four-element vectors and records 2 through 4 as
one-element parameters. This carrier is split into four row groups; npunlock
selects all four only when the complete source and descriptor pattern matches.
The [hybrid five-input example](../examples/example_hybrid_five_input_custom.py)
implements residual addition, per-row affine scaling, and an upper clamp.
Neither recipe establishes other vector/scalar mixtures, shapes, positions,
or precisions.

Use independent, non-symmetric test values so the oracle detects swapped or
ignored inputs. The complete [multi-layer two-input
example](../examples/example_multilayer_multi_input.py) places this kernel
between ordinary graph operations and checks the result.

For two-input elementwise carriers, `custom()` infers the symbolic NumPy-style
broadcast output shape. This is shape convenience, not general carrier
support: only the exact layouts listed above are validated. A one-element ACT
input must be read at index zero; the validated `(4,8)` plus `(1,8)` row
carrier instead presents equal eight-element records and uses `input_1[i]`.
Follow the
[scalar-input example](../examples/example_branched_network.py) or
[per-row scalar example](../examples/example_per_row_scalar_custom.py) or
[FP16 row-broadcast example](../examples/example_broadcast_custom.py) or
[FP32 row-broadcast example](../examples/example_f32_broadcast_custom.py) or
[unsigned broadcast example](../examples/example_unsigned_broadcast_custom.py)
exactly before generalizing a layout.

## 8. Inspect and bind a carrier explicitly

`npu.compile()` is convenient when positional mapping is unambiguous. Use
`npu.prepare()` when a graph has several custom operations, conversions, DPU
work, fusion, or any placement you want to review:

```python
prepared = npu.prepare(graph)
print(prepared.explain())

binary_f16 = prepared.find_group(input_count=2, dtype="f16")
plan = prepared.plan(bindings={custom_output: binary_f16})
print(plan.explain())

program = plan.build()
```

An `ActGroup` describes the actual compiler output: arity, per-tensor dtype,
invocation/range indices, element counts, byte spans, and observed layout
flags. `find_group()` requires exactly one match and reports ambiguity instead
of guessing.

For unequal binary inputs, `group.scalar_input_indices` reports the stable
one-element positions. Use a filter such as
`prepared.find_group(scalar_input_indices=(0,))` when operand position matters.

The binding key is the symbolic output returned by `npu.custom()`, not a string
node name. Native graph blobs do not provide a proven general mapping from an
IR source name to an ACT range.

For porting experiments, a prepared graph can compile one controlled carrier
substitution without changing the original graph:

```python
probe = prepared.probe_carrier(custom_output, "Sqrt")
print(probe.accepted, probe.group_index, probe.comparison.mismatch_names)
```

This preserves the node's existing inputs, outputs, attributes, weights, and
build flags. Treat an accepted result as differential evidence only; it is not
used as a patch binding automatically. If you already have both native blobs,
`npu.compare_native_graphs(baseline, candidate)` applies the same strict
comparison in memory. See [Porting to Linux and newer NPUs](PORTING.md) for the
workflow and its limits.

For a stronger manual review, try several substitutions once and retain both
the comparisons and any compiler/inspector failures:

```python
report = prepared.probe_candidates(custom_output, ("Sqrt", "Tanh", "Relu"))
print(report.explain())
evidence = report.evidence
```

This reports `consistent` only when at least two accepted probes agree on one
group. A failed candidate remains in `report.failures` and never counts as
evidence. It still does not change the prepared graph's bindings.

For a group that writes directly to a graph output, you can separately review
the routing/replica coverage proof:

```python
coverage = prepared.inspect_group_completeness()
if evidence.group_index is not None:
    print(coverage[evidence.group_index].state)
```

`not_observed` is expected for many intermediate groups; it does not mean
complete. The [porting guide](PORTING.md) defines the three states and their
limits.

To inspect intermediate groups without collapsing different kinds of evidence,
use the broader read-only view:

```python
for group in prepared.inspect_group_coverage():
    print(group.group_index, group.complete_kinds)
```

Each record keeps `direct_output_state`, `act_consumer_state`,
`dpu_consumer_state`, `dma_dpu_consumer_state`, and
`dma_act_consumer_state` independent. The DMA/DPU kind is the narrow
multi-tile route where ACT outputs form an exact DMA fan-in and translated
multicast destinations uniquely match DPU inputs. The DMA/ACT kind instead
requires an exact ACT-output spill to DDR followed by scheduled reloads that
cover unique inputs of a later ACT group. A complete relationship proves
bounded native invocation coverage, not source identity, and is not an
automatic patch binding.

The compact helper runs the probes and combines them with coverage only when
the identified group has exactly one complete route meaning:

```python
reviewed = prepared.probe_binding(
    custom_output,
    report,
)
program = prepared.build(bindings={custom_output: reviewed})
```

`probe_binding()` rejects zero or several complete kinds. When several route
meanings are complete, use `probe_mapping()` and `inspect_group_coverage()` to
inspect them, then name the intended `direct_output`, `act_consumer`,
`dpu_consumer`, `dma_dpu_consumer`, or `dma_act_consumer` kind explicitly. The
helper records the exact prepared native-blob hash, carrier comparisons, and
coverage record; planning rechecks them. This opt-in path does not change
ordinary positional mapping, and raw explicit `ActGroup` bindings remain
available.
Passing the report avoids recompiling its successful candidates.

For example, a custom elementwise operation immediately before a DPU
`MatMul` may use the DMA-mediated form:

```python
reviewed = prepared.probe_binding(
    custom_output,
    ("Minimum", "Subtract"),
    coverage_kind="dma_dpu_consumer",
)
```

This succeeds only when both controlled carrier substitutions identify the
same ACT group and every observed tile has one uniquely translated DPU input.
The [branched DPU/ACT example](../examples/example_branched_network.py) applies
this workflow to three custom nodes in one graph. It deliberately uses
different candidate lists for the scalar-input, full binary, and FP32 unary
nodes because carrier lowering depends on arity, dtype, shape, and graph
context; there is no universal fallback list. The example also saves the
reviewed plan as a prepared bundle and rebuilds from the restored evidence, so
the controlled graph compilations are a one-time review cost.

For several kernels, filter groups with independent fields and bind them all:

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

See the [mixed-precision multi-custom
example](../examples/example_mixed_precision_multi_custom.py) for that exact
workflow. The [branched DPU/ACT
example](../examples/example_branched_network.py) shows the reviewed-probe path
for a graph where automatic positional mapping intentionally refuses to guess.

## 9. Write a conversion kernel

Conversion carriers have different input and output pointer types. They always
use explicit prepared-group binding. This U8-to-FP32 kernel applies an affine
transform while changing representation:

```c
#include <npunlock/npu3720_kernel.h>

void controlled_act(unsigned layerParams) {
    act_abi_invocation invocation;
    ACT_ABI_LOAD_INVOCATION32_OR_RETURN(layerParams, invocation);
    const unsigned char *input =
        ACT_ABI_INPUT_PTR32(const unsigned char, invocation, 0u);
    float *output = ACT_ABI_OUTPUT_PTR32(float, invocation, 1u);

    for (unsigned i = 0; i < invocation.element_count; ++i) {
        output[i] = (float)input[i] * 0.5f - 7.25f;
    }
}
```

`destination_type` gives both the carrier and the symbolic custom output its
FP32 dtype; no private `_dtype` argument is needed:

```python
x = npu.input("x", shape=(1, 32), dtype="u8")
y = npu.custom(
    x,
    source=kernel_c,
    carrier="Convert",
    destination_type="f32",
    _name="y",
)

prepared = npu.prepare(npu.Graph([x], [y], name="u8_to_f32"))
group = prepared.find_group(input_dtypes=("u8",), output_dtype="f32")
program = prepared.build(bindings={y: group})
```

Patch the complete discovered group. The two invocations in the tested
conversion carriers are full-tensor tile replicas, not two output partitions.
The runnable [U8-to-FP32 example](../examples/example_u8_to_f32_custom.py) and
[connected FP32/FP16 conversion
example](../examples/example_conversion_kernels.py) cover all three validated
directions.

## 10. Use math functions and structured C carefully

Most familiar single-precision `libm` names are present in MoviTools'
`mlibm.a`, but only the documented functions have been executed against host
oracles. For the tested math path, enable its link-compatibility definitions
before including the header:

```c
#define MLIBM_DEFINE_LINK_COMPAT 1
#include <npunlock/npu3720_kernel.h>
```

The execution-tested set includes `sinf`, `cosf`, `logf`, `erff`, `atanf`,
`expf`, `sqrtf`, and `tanhf`; individual success does not guarantee that every
large combination will execute. You normally call these names without
including `<math.h>`. The [symbol inventory](../ABI/MLIBM_SYMBOLS.md) contains
other candidates whose signatures and hardware behavior still need validation.

Bounded branches, loops, same-file helpers, function pointers, local scratch,
and invocation-local reductions have examples. Dynamic allocation, recursion,
cross-invocation access, nonempty kernel data sections, and unresolved symbols
remain unsupported. Start with [Writing custom kernels](CUSTOM_KERNELS.md)
before moving beyond an elementwise loop.

## 11. Reuse compiled work

Compile C once and save the validated SHAVE ELF:

```python
compiled = npu.compile_kernel(kernel_c)
compiled.save("affine.elf")

x = npu.input("x", shape=(1, 32), dtype="f16")
y = npu.custom(x, kernel=compiled, carrier="Abs", _name="y")
program = npu.compile(npu.Graph([x], [y]))
```

A later process can use `npu.CompiledKernel("affine.elf")`. If every custom
node uses a compiled ELF, patching no longer needs MoviTools, though graph
compilation still uses the Intel driver.

Export the final executable program when deployment should need neither C
source nor MoviTools:

```python
program.export("affine.npunlock")

loaded = npu.load("affine.npunlock")
output = loaded.run_one(input_value)
```

The bundle records tensor contracts, graph/compiler provenance, hashes, and
patch reports. Execution still requires compatible NPU hardware and driver.
Use `PreparedGraph.export()` instead when you want to preserve a reviewed
carrier and binding while continuing to rebuild the C kernel. When a plan uses
`ReviewedBinding`, the prepared bundle also carries the controlled candidate
native blobs. Loading reruns their strict comparisons and the selected coverage
check without recompiling the graph.

## 12. Use shared buffers when repeated copies matter

Ordinary `Program.run()` copies NumPy inputs through internal Level Zero
allocations. For repeated inference, allocate NumPy-compatible host/NPU shared
arrays once:

```python
shared_input = program.shared_input()
shared_output = program.shared_output()

np.copyto(shared_input, input_value)
program.run({"x": shared_input}, outputs={"y": shared_output})
np.testing.assert_array_equal(shared_output, expected)
```

Shared arrays are tied to the program's native session. Keep them alive only
with that session, and do not bind an array created by another program.

## 13. Diagnose failures in the right layer

Use this order when a new kernel does not work:

1. Run `python -m npunlock doctor --full` to separate driver, native runtime,
   MoviTools, and end-to-end failures.
2. Confirm the carrier recipe matches arity, dtype, and shape exactly.
3. Call `prepared.explain()` and inspect the discovered groups.
4. Call `plan.explain()` before compiling C or patching a complex graph.
5. If linking reports undefined math symbols, check the math feature switch and
   the [symbol inventory](../ABI/MLIBM_SYMBOLS.md).
6. If execution succeeds but values are wrong, compare every element and make
   the oracle invocation-local where the algorithm is invocation-local.
7. Reduce to the closest runnable example before trying a new carrier.

Do not bypass an unsupported or ambiguous contract with guessed invocation or
range indices. The failure is often telling you that the compiler produced a
different graph structure, not that the C expression itself is invalid.

## Where to continue

- [Writing custom kernels](CUSTOM_KERNELS.md) documents helpers, local memory,
  math, control flow, and the measured ABI boundary.
- [Python API](PYTHON_API.md) covers graph construction, prepared plans,
  bundles, raw blobs, and execution APIs in detail.
- [Current limitations](LIMITATIONS.md) is the authoritative compatibility
  boundary.
- [How npunlock works](HOW_NPUNLOCK_WORKS.md) explains the graph/compiler/
  patching pipeline.
- [Runnable examples](../examples) provide host-oracle-backed recipes for each
  supported custom-kernel family.
- [XGBoost contribution](../contrib/xgboost/README.md) shows a deliberately
  narrow tree-ensemble workload, including a two-part runtime forest whose two
  NPU traversals are connected inside one graph: the first ACT node forwards
  the original features and partial margins directly to the second. Its
  `XGBoostRuntime` facade automatically selects the validated single- or
  connected two-stage backend. The runtime-loaded paths shard samples across
  four independent 4 KiB invocation-local spans before chunking larger host
  batches, providing one-execution capacities of 136 samples for the
  single-stage 30-feature model and 128 for the connected model. This does not
  assume cross-invocation communication or enlarge a local span. Both paths use
  one graph execution per host chunk; the two-node path is a narrow positional
  contract, not a claim of arbitrary graph fusion. Pass an explicit
  `cache_dir` to reuse a model-independent compiled program across processes;
  each hit rechecks live driver/compiler provenance, MoviTools and native
  runtime file identity, bundle integrity, tensor contracts, and the
  patch-report hash chain before execution. The runtime forest remains a
  separately uploaded input and is not written to that cache.
- [HNSW contribution](../contrib/hnsw/README.md) runs hierarchical greedy
  descent and dynamic best-first base-layer traversal entirely inside an
  ACT-SHAVE kernel. Its throughput carrier uses four execution-tested
  12,288-element FP32 invocation spans, subdivides each into twelve independent
  1,024-element query/index chunks, and therefore runs up to 48 queries per
  submission. This exact 48 KiB local span is validated only for the documented
  FP32 `Subtract` shape on NPU3720/compiler 8.3. The bounded 48-node index is
  duplicated into every chunk; the implementation does not assume graph-global
  reads or cross-invocation communication. Its
  persistent program cache validates kernel, runtime, driver/compiler,
  MoviTools, bundle, tensor, and patch provenance before reuse. The real-data
  digits/PCA demo checks node IDs, raw FP32 distances, and traversal counters
  against a host oracle and compares recall with hnswlib. The backend uploads
  the repeated graph/vector payload once, then updates only per-query slices in
  shared Level Zero memory for subsequent submissions. Its 16-dimensional
  fast path and host oracle use the same fixed four-lane FP32 reduction tree;
  this is a validated narrow arithmetic contract, not reassociation-invariant
  distance arithmetic.

[Back to documentation index](README.md)
