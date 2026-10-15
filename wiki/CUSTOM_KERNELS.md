# Custom kernels

[Documentation index](README.md)

Custom kernels are C functions compiled for the NPU3720 ACT-SHAVE processors.
They run inside an invocation environment created by Intel's graph compiler.
That environment supplies tensor pointers and dimensions; the kernel supplies
the computation.

The contract described here is intentionally narrow and execution-tested. It
is not a general Intel SHAVE ABI.

## Entry point

Start NPU3720 kernel source with the bundled target header, then define the
default entry function:

```c
#include <npunlock/npu3720_kernel.h>

void controlled_act(unsigned layerParams);
```

`shavecc` recognizes that exact include as the first non-whitespace directive,
or immediately after the math feature switch documented below, and expands it
from bytes embedded in `npunlock.dll`. No temporary file or installed include
search path is involved. The physical header is also installed by the native
package and included in Python wheels for source browsing and editor support.

The filename carries the NPU generation namespace. The header encodes the
observed NPU3720 ACT layout, including low-32-bit pointers and 0x28-byte tensor
records. It is not a portable SHAVE or future-NPU ABI.

The main helpers are:

- `act_abi_load_le_u32()` for raw little-endian fields;
- `ACT_ABI_LOAD_INVOCATION32_OR_RETURN()` for guarded rank,
  dimensions, and element-count loading;
- `act_abi_dimension()` for a bounds-checked invocation dimension lookup;
- `ACT_ABI_INPUT_PTR32()` and `ACT_ABI_OUTPUT_PTR32()` for tensor addresses;
  and
- `MLIBM_DEFINE_LINK_COMPAT` as the opt-in switch for link-compatibility
  symbols required by the observed `mlibm.a` math path.

`layerParams` is the low 32-bit address of the current invocation's parameter
block. The header makes that assumption visible but does not turn it into a
cross-generation contract.

On the tested host-visible FP16 and FP32 unary paths, runtime self-check
kernels observed zero in the upper 32-bit word of both input and output data
addresses. They also re-read the static, rank, type, placement, dimension, and
stride fields that `patchblob` had validated before installation. This supports
the header's `PTR32` accessors for NPU3720; it does not support nonzero upper
halves, another NPU generation, or a parameter-block address wider than the
entry's 32-bit argument.

The dimensions array contains one 32-bit value per rank dimension. Always
validate rank and dimension multiplication before entering the element loop.

## Unary FP16 example

This complete kernel adds one to every element in its invocation-local chunk:

```c
#include <npunlock/npu3720_kernel.h>

void controlled_act(unsigned layerParams) {
    act_abi_invocation invocation;
    ACT_ABI_LOAD_INVOCATION32_OR_RETURN(layerParams, invocation);
    const __fp16 *input =
        ACT_ABI_INPUT_PTR32(const __fp16, invocation, 0u);
    __fp16 *output = ACT_ABI_OUTPUT_PTR32(__fp16, invocation, 1u);

    for (unsigned i = 0; i < invocation.element_count; ++i) {
        output[i] = (__fp16)((float)input[i] + 1.0f);
    }
}
```

The capacity guard belongs to the kernel source. Choose it for the exact
validated carrier; do not silently assume that every invocation has the same
size.

## Tensor descriptor positions

The validated parameter records are 0x28 bytes each:

```text
unary:
  input  at layerParams + 0x00
  output at layerParams + 0x28

binary:
  input A at layerParams + 0x00
  input B at layerParams + 0x28
  output  at layerParams + 0x50
```

Within each record, the fields used by the examples are:

```text
+0x00  low 32 bits of data address
+0x04  observed-zero upper 32 bits of data address
+0x0c  rank
+0x10  low 32 bits of dimensions-array address
```

`patchblob` validates the rest of the observed contract before installation:
static dimensions, dense strides, supported precision, declared element counts
and spans, CMX placement, and a disjoint output.

## Supported precision

Static dense FP16 is the primary custom-kernel contract. Both unary and narrow
two-input layouts have been executed successfully.

FP32 custom execution includes the validated unary accuracy-mode path and one
same-precision two-input carrier with a scalar second input. The Python
frontend adds the required precision-preservation metadata and compiler flag
automatically for an FP32 custom node. This does not establish general FP32
carriers or multi-input support.

I32 custom execution includes one same-precision unary `Abs` carrier on graph
compiler 8.3. Its four invocations each expose 16 signed 32-bit input and
output elements. A custom `3*x + 7` kernel matched all 32 graph outputs exactly.
Use `int` pointers with the same NPU3720 address helpers. This does not establish
I32 binary carriers, mixed integer contracts, overflow behavior, or other
integer precisions.

Although the symbolic IR serializer accepts additional data-type names, that
does not make them valid custom-kernel or `Program.run()` contracts.

## Mixed-precision conversion kernels

The version 2 target ABI supports the two execution-tested unary conversion
layouts: dense FP32 input to FP16 output and dense FP16 input to FP32 output,
with equal element counts. A conversion kernel uses the ordinary descriptor
positions but gives the pointers different C types:

```c
const float *input = ACT_ABI_INPUT_PTR32(const float, invocation, 0u);
__fp16 *output = ACT_ABI_OUTPUT_PTR32(__fp16, invocation, 1u);
```

The reverse direction uses `const __fp16 *` and `float *`. The graph compiler
emits two full-tensor tile replicas for each tested conversion group, so select
the complete discovered group; do not interpret the two targets as output
partitions.

Python deliberately requires explicit prepared-graph binding for these
groups. Filter on the independently validated input and output contracts:

```python
prepared = npu.prepare(graph)
to_f16_group = prepared.find_group(
    input_dtypes=("f32",), output_dtype="f16"
)
program = prepared.build(bindings={custom_output: to_f16_group})
```

The native C interface represents the same information with
`patchblob_tensor_contract_v2`, `patchblob_target_v2`,
`patchblob_discover_targets_v2()`, and `patchblob_patch_v2()`. The original
target ABI remains available and still rejects unequal input/output
precisions. Version 2 does not imply arbitrary mixed input types, element-count
changes, broadcasting, or automatic conversion insertion.

See the complete
[connected conversion example](../examples/example_conversion_kernels.py),
which runs custom conversions in both directions around an ordinary FP16
operation and checks the final FP32 tensor against NumPy.

## Invocation-local memory

The Intel compiler partitions tensors into ACT invocations. The descriptor
seen by the kernel describes the current chunk, not necessarily the full
graph tensor.

Validated FP16 carriers have advertised chunks from 8 through 2048 elements.
A kernel has read the full 4096-byte span of a 2048-element chunk. This is a
confirmed lower bound for invocation-local reach, not a hardware maximum and
not permission to access another invocation's memory.

Operations requiring neighboring values must define behavior at each chunk
boundary. Graph-global coordinates and cross-chunk halos are not available in
the current contract.

Graph-level shape is not passed through verbatim. With graph compiler 8.3, a
logical FP16 tensor shaped `(4, 8)` on an `Abs` carrier produced four
invocations. Each entry reported rank 4 and dimensions `(8, 1, 1, 1)`, with
an element count of 8. A validated entry checked all four dimensions, derived
its loop width from `act_abi_dimension(&invocation, 0)`, and reversed each
host-visible row exactly through copied and shared execution.

See the [dimension-aware example](../examples/example_dimension_aware.py).
Always validate the runtime invocation rank/dimensions required by the kernel;
do not use the symbolic graph shape as an unverified device-side layout.

## Control flow, helper calls, and local storage

A custom entry does not have to be one flat elementwise loop. One validated
FP16 kernel uses all of the following together:

- a same-file `static` helper marked `__attribute__((noinline))`;
- a branch inside that helper;
- next-element reads clamped at each invocation boundary; and
- a source-level `volatile float scratch[32]` local array.

The helper remained a distinct function in the linked ELF, the ELF contained
no unresolved symbols or relocations, and copied and shared execution both
matched an exact NumPy oracle. See the complete
[structured-kernel example](../examples/example_complex_kernel.py).

This result validates bounded control flow, one internal helper, and that
specific 128-byte source-level local array. It does not establish a general
stack limit or calling convention. Keep helper definitions in the same source
buffer, bound every local allocation, and treat recursion, cross-invocation
access, and large stack objects as unsupported.

A second validated kernel extends that result to a three-helper nested call
chain, a local context structure, a source-level
`volatile float scratch[64]`, and a five-tap loop over an embedded 20-byte
coefficient table. Its clamped neighborhood is deliberately local to each
32-element invocation chunk. Copied and shared execution matched the same
chunk-aware host calculation exactly. See the
[window-kernel example](../examples/example_window_kernel.py).

A dedicated single-entry probe later wrote and hashed every byte of a
`volatile unsigned char` array. The exact tested body repeatedly produced
correct output through 7,168 source-level bytes. At 7,184 bytes, one run was
correct and four silently corrupted the second 16-element half of a 32-element
result. At 7,200 bytes, observed outcomes included correct output, corruption,
and a device error; at 8,192 bytes, the NPU returned a device error. This is an
unstable boundary for that compiler, carrier, and body—not a public stack
allowance. Frame overhead, helper calls, compiler changes, and scheduling may
reduce the available space. Keep local objects much smaller and validate
results against a semantic oracle; successful inference alone cannot detect
this overflow.

The example declares the coefficients `const volatile`: `const` prevents the
kernel from modifying them, while `volatile` keeps MoviTools from folding the
entire table into instructions. The default linker places the object inside
the final `.text` image. On the tested toolchain that combined section is
marked writable and executable, so “constant” describes the C contract rather
than an ELF memory-protection guarantee.

One bounded runtime function-pointer pattern is also validated. An entry
selects between two same-file non-inlined transforms from each input value,
stores the selection in a volatile local function pointer, and passes it to a
separate non-inlined dispatcher. Both targets and the dispatcher remained
distinct ELF symbols; the final image had no relocations or undefined runtime
helpers, and copied/shared results matched exactly.

See the [function-dispatch example](../examples/example_function_dispatch.py).
The [function-table example](../examples/example_function_table.py) extends
this to three same-file targets selected by a runtime table index. The table is
an explicitly initialized `volatile` invocation-local array. That distinction
matters: the tested compiler placed both a file-scope constant table and a
local constant initializer in nonempty `.arg.data`, which `shavecc` correctly
rejected. Runtime assignment kept every target address in the executable image
and invocation-local storage; the linked ELF had no relocations or undefined
symbols.

This does not validate global or statically initialized function-pointer
tables, pointers loaded from tensor data, external targets, callbacks,
recursion, or arbitrary indirect call graphs.

## Invocation-local reductions

Multiple passes and reductions over the current invocation are also usable. A
validated softmax-like kernel:

1. finds the maximum FP16 input value as an FP32 scalar;
2. evaluates `expf(input[i] - maximum)` into local scratch;
3. accumulates the FP32 exponential sum; and
4. writes normalized FP16 results in a third pass.

For the tested logical 64-element tensor, the carrier exposed 32-element
invocation chunks. The kernel therefore computes two independent normalized
chunks; it is not graph-global or axis-wide softmax. Copied and shared results
matched a NumPy oracle with `rtol=5e-3` and `atol=5e-4`. See the
[invocation-local softmax example](../examples/example_invocation_softmax.py).

Linking `expf` produced a 53,796-byte ELF because retained non-loadable debug
sections dominate the file. The executable `.text` load image was 976 bytes.
`patchblob` installs the validated load image, not the complete ELF container.

## Data-dependent loops and local permutation

Loop trip counts may depend on invocation data in a bounded kernel. A
validated insertion-sort entry copies one FP16 invocation into FP32 scratch,
uses an inner `while` loop to shift values through indirectly indexed scratch
positions, and writes the sorted values back. For the tested logical
64-element tensor, copied and shared execution exactly matched two independent
32-element NumPy sorts.

See the [invocation-local sort example](../examples/example_invocation_sort.py).
The example establishes data-dependent local control flow and value
permutation. It does not establish graph-global sorting, cross-invocation
movement, index outputs, unbounded loops, or dynamic allocation.

## Carrier selection

Python custom nodes name a carrier explicitly:

```python
y = npu.custom(
    x,
    source=kernel_c,
    carrier="Abs",
    _name="y",
)
```

For a single-output custom operation, omitted `_shape` and `_dtype` values are
inherited from the first input. Supply either override only when that part of
the output contract differs and the carrier layout has been independently
validated, or use `_outputs` to describe multiple outputs. The established
automatic path normally remains same-shape and same-dtype; the scalar-second-
input exception is described below.

The carrier's compiled arity and tensor layout must match the C entry. `Abs`
is used by the validated unary examples. `Maximum` is used by the validated
two-input carriers.

Two narrow broadcasting cases are execution-tested. The unequal-count layout
has an ordinary first input and output plus a one-element second input of the
same precision. Its C loop still uses the output chunk's
`invocation.element_count`, but reads the second input only at index zero:

```c
for (unsigned i = 0; i < invocation.element_count; ++i) {
    output[i] = input_0[i] * scale + input_1[0];
}
```

Do not index the scalar with `i`.

Separately, graph compiler 8.3 presents one FP16 row broadcast as dense ACT
operands for a `Maximum` carrier. Host inputs shaped `(4, 8)` and `(1, 8)`
produced four ACT invocations whose contracts were `(8, 8) -> 8`: both input
records were dense and equal-sized inside every invocation. The custom C
therefore reads `input_1[i]`, not `input_1[0]`. This establishes the descriptor
contract visible to the entry, not whether backing storage is physically
copied or reused. It validates that exact carrier and shape, not arbitrary
NumPy broadcasting or an unequal ACT descriptor.

Call `prepared.explain()` before patching to see stable per-invocation element
counts such as `per-invocation=(8,8)->8`. The same values are available as
`ActGroup.per_invocation_input_element_counts` and
`ActGroup.per_invocation_output_element_count`. A `None` value means the
counts vary across the group's invocations and must be inspected target by
target.

One five-input FP16 `FakeQuantize` carrier is also validated. Its parameter
records contain inputs 0 through 4 followed by output record 5. Four observed
invocations each reported `(8,8,8,8,8) -> 8`, so the entry uses:

```c
const __fp16 *input_4 = ACT_ABI_INPUT_PTR32(const __fp16, invocation, 4u);
__fp16 *output = ACT_ABI_OUTPUT_PTR32(__fp16, invocation, 5u);
```

All five inputs were independently host-bound and semantically distinguished.
This is one validated carrier contract, not support for arbitrary arity.

The same carrier also runs a composed multi-pass kernel using all five inputs:
data, residual, scale, bias, and gate. The entry adds data and residual into
FP32 scratch, accumulates an invocation-local sum of squares, calls `sqrtf` for
RMS normalization, then applies per-element scale, bias, and gating. Four
observed invocations each retained the `(8,8,8,8,8) -> 8` contract. Copied and
shared execution matched a chunk-aware NumPy oracle within `rtol=5e-3` and
`atol=5e-4`, with zero observed FP16 error for the deterministic proof input.

See the [fused RMS-normalization example](../examples/example_fused_rmsnorm.py).
This composes several validated mechanisms in one entry; it is not a claim of
standard graph-global RMSNorm semantics or arbitrary five-input layouts.

Automatic positional mapping normally maps one computational node to one ACT
group. A large custom operation may use several consecutive compatible groups
when their targets uniquely and exactly cover its declared output tensor. See
[How npunlock works](HOW_NPUNLOCK_WORKS.md#4-find-a-compatible-act-group).

## Math functions and linked data

MoviTools' `mlibm.a` supplies functions such as `tanhf`. `shavecc` links the
archive with section garbage collection so only reachable code remains.

The tested MoviTools compiler and `mlibm.a` combination accepts familiar
function names without including `<math.h>`. Hardware-oracle probes currently
cover the FP32 unary functions `sinf`, `cosf`, `logf`, `erff`, `atanf`,
`expf`, `sqrtf`, and `tanhf`. The GELU examples use `tanhf`, the
invocation-local softmax example uses `expf`, and the fused RMS example uses
`sqrtf`. A host platform's `libm` is not the contract: any other archive name
is only a candidate until MoviTools accepts its inferred prototype, the
archive resolves it, the linked ELF passes validation, and execution matches
a host oracle.

The [observed `mlibm.a` symbol inventory](../ABI/MLIBM_SYMBOLS.md) lists the
names that may be referenced. The archive does not determine their C
signatures. A likely signature can be inferred from the conventional function
name or an existing `libm` implementation, but it must be treated as inferred
until compilation and execution against a host oracle confirm it.

The observed archive also leaves references to `strtof`, `__truncdfsf2`, and
`__fixsfdi`. A kernel using this math path should enable the bundled definitions
before including the header:

```c
#define MLIBM_DEFINE_LINK_COMPAT 1
#include <npunlock/npu3720_kernel.h>
```

The macro supplies the observed symbol signatures, two deterministic link-only
stubs, and the tested binary32-to-integer helper. Do not call `strtof` or
`__truncdfsf2`: their bundled bodies return zero and exist only to close
otherwise unused archive references. They are known to be harmless only for
the execution-tested functions above. These are NPU3720/MoviTools
link-compatibility definitions, not a general C runtime.

Composition also matters. A probe combining `sinf`, `cosf`, `logf`, and
`erff` in one runtime-selected kernel executed correctly with a 3,056-byte
image. Adding `atanf` produced a 3,504-byte image that passed ELF and patch
validation but faulted during NPU execution, even though the carrier's
original code extent was 5,280 bytes. This is not a proven universal size
threshold; keep complex math closures small and validate each combined kernel
on hardware.

The resulting kernel must be self-contained in `.text`. The current validator
requires `.arg.data` to be empty, so mutable globals, retained writable library
state, unresolved relocations, and general helper runtimes are unsupported.
Source-level constant objects are usable when the linker places them inside
the final `.text` image. The window-kernel example demonstrates a retained
constant table; the tested linker marks the combined output section writable
and executable even though the C object is `const`, so ELF memory protection
is not part of this contract.

## Compiler definitions and linker script

`npu.compile(..., definitions=("NAME=123",))` forwards a constrained set of
uppercase decimal definitions. The default NPU3720 linker script is embedded
in the native library. Supplying another script is an advanced override and
does not relax ELF validation.

For a graph with different settings per kernel, create a prepared build plan
and override only the relevant symbolic outputs:

```python
plan = prepared.plan(
    bindings={custom_output: selected_group},
    kernels={
        custom_output: npu.KernelSpec(
            "kernels/custom.c",
            definitions=("MODE=2",),
        )
    },
)
print(plan.explain())
program = plan.build()
```

`KernelSpec` reads source and linker paths when the plan is created, so the
hashes printed by `explain()` describe the bytes that will actually be sent to
MoviTools. Identical specifications in one plan reuse their compiled ELF.

To reuse an already linked ELF, replace that override with
`npu.CompiledKernel("kernels/custom.elf")`. The object snapshots the file and
`explain()` reports its SHA-256. A plan in which every custom output has a
`CompiledKernel` override can build without MoviTools; a mixed plan compiles
only its `KernelSpec` entries. The ELF must still be a supported linked
`3720xx` image. The normal native patch path validates it before changing the
carrier, so this option does not relax the kernel ABI or graph contract.

Compile once and save the linked image through the same high-level API:

```python
compiled = npu.compile_kernel(
    "kernels/custom.c",
    definitions=("MODE=2",),
)
compiled.save("kernels/custom.elf")
```

Passing an existing `KernelSpec` is equivalent and preserves its snapshotted
settings; separate definitions or a linker script cannot override that spec.
The returned object can be passed directly to `PreparedGraph.plan()`. A later
process can reconstruct the same value with
`npu.CompiledKernel("kernels/custom.elf")`.

A graph intended for compiled-kernel deployment can attach the immutable ELF
directly to its symbolic node:

```python
custom_output = npu.custom(
    x, kernel=compiled, carrier="Abs", _name="custom_output"
)
program = npu.compile(npu.Graph([x], [custom_output]))
```

`kernel=` accepts either `KernelSpec` or `CompiledKernel`, and cannot be
combined with `source=`. A source-free node remains useful when the kernel is
selected later: it requires an explicit `KernelSpec` or `CompiledKernel`
override through `kernels={output: ...}`. Calling `plan()` without one is
rejected before MoviTools or graph mutation. Supplying `source=` remains the
normal compact workflow for plans that compile C during the build.
The one-call form requires an unambiguous automatic mapping. Use `prepare()`
and `prepared.plan(kernels=...)` when you want to inspect `plan.explain()`
before patching or when the graph needs explicit bindings.

When iterating on kernel C, save the driver-compiled carrier and reviewed
binding separately with `prepared.export("network.prepared.npunlock",
plan=plan)`. A later process can call `npu.load_prepared(..., graph=graph)`,
which requires the same serialized graph but deliberately accepts changed
kernel source. Prepared bundles do not embed source or compiled ELF inputs;
provide `KernelSpec` or `CompiledKernel` again after loading. See
[Python API](PYTHON_API.md#reuse-a-prepared-carrier).

## Runnable examples

- [FP16 GELU](../examples/example_gelu.py)
- [FP32 GELU](../examples/example_gelu_f32.py)
- [Unary I32 affine kernel](../examples/example_i32_custom.py)
- [Multi-layer, two-input custom kernel](../examples/example_multilayer_multi_input.py)
- [Two-input custom kernel with a descriptor-level row broadcast](../examples/example_broadcast_custom.py)
- [Five-input custom kernel](../examples/example_five_input_custom.py)
- [Fused five-input invocation-local RMS normalization](../examples/example_fused_rmsnorm.py)
- [Dimension-aware local transform](../examples/example_dimension_aware.py)
- [Runtime dispatch through a bounded same-file function pointer](../examples/example_function_dispatch.py)
- [Runtime-indexed invocation-local function table](../examples/example_function_table.py)
- [Structured kernel with a helper, branching, neighbors, and local scratch](../examples/example_complex_kernel.py)
- [Five-tap window kernel with nested helpers and an embedded coefficient table](../examples/example_window_kernel.py)
- [Invocation-local softmax with max/sum reductions and `expf`](../examples/example_invocation_softmax.py)
- [Invocation-local insertion sort with a data-dependent inner loop](../examples/example_invocation_sort.py)
- [Mixed-precision graph with FP32 unary and FP16 binary custom branches](../examples/example_mixed_precision_multi_custom.py)
- [Branched DPU/ACT network with three custom kernels](../examples/example_branched_network.py)
- [Connected mixed-precision custom/DPU composition](../examples/example_connected_mixed_precision.py)

Each example includes a NumPy reference. Successful graph submission is not a
correctness oracle; use equivalent host validation for every new kernel.

The branched example deliberately uses three distinguishable contracts: FP32
unary, FP16 scalar-second-input, and full FP16 binary. Its carrier has fewer
ACT groups than symbolic computational nodes because `MatMul` is lowered to
DPU work, so automatic positional mapping refuses the graph. The example uses
`prepare()`, filters the validated group contracts, and explicitly binds each
custom output before patching.

On NPU3720 with graph compiler 8.3, this graph produced six ACT groups for
seven computational nodes alongside nonempty DPU invariant and variant
sections. Its connected FP16 result and independent FP32 result both matched
the NumPy oracle exactly. This is one composition proof, not a claim that
arbitrary branched graphs or matrix operations are supported.

The connected mixed-precision example goes further along one data path: a
custom FP32-to-FP16 conversion feeds a two-input FP16 kernel, an ordinary DPU
`MatMul`, and a custom FP16-to-FP32 conversion. It reviews and saves a build
plan, reloads the unpatched carrier, rebuilds all kernels, exports the final
program, reloads it, and checks every execution against an exact NumPy oracle.

For the complete byte-level entry and ELF contract, see
[ACT kernel ELF ABI](../ABI/KERNEL_ELF.md).

[Back to documentation index](README.md)
