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
- `ACT_ABI_INPUT_PTR32()` and `ACT_ABI_OUTPUT_PTR32()` for tensor addresses;
  and
- `MLIBM_DEFINE_LINK_COMPAT` as the opt-in switch for link-compatibility
  symbols required by the observed `mlibm.a` math path.

`layerParams` is the low 32-bit address of the current invocation's parameter
block. The header makes that assumption visible but does not turn it into a
cross-generation contract.

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
+0x0c  rank
+0x10  low 32 bits of dimensions-array address
```

`patchblob` validates the rest of the observed contract before installation:
static dimensions, dense strides, supported precision, matching element
counts and spans, CMX placement, and a disjoint output.

## Supported precision

Static dense FP16 is the primary custom-kernel contract. Both unary and a
narrow two-input layout have been executed successfully.

FP32 custom execution is limited to the validated unary accuracy-mode path.
The Python frontend adds the required precision-preservation metadata and
compiler flag automatically for an FP32 custom node. This does not establish
general FP32 carriers or FP32 multi-input support.

Although the symbolic IR serializer accepts additional data-type names, that
does not make them valid custom-kernel or `Program.run()` contracts.

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
automatic path remains same-shape and same-dtype.

The carrier's compiled arity and tensor layout must match the C entry. `Abs`
is used by the validated unary examples. The validated two-input example uses
`Maximum` for one exact static FP16 graph.

Automatic positional mapping is accepted only when the complete graph maps
one-to-one to validated ACT groups. See
[How npunlock works](HOW_NPUNLOCK_WORKS.md#4-find-a-compatible-act-group).

## Math functions and linked data

MoviTools' `mlibm.a` supplies functions such as `tanhf`. `shavecc` links the
archive with section garbage collection so only reachable code remains.

The tested MoviTools compiler and `mlibm.a` combination allows most
conventional `libm` functions to be called by their normal C names without
including `<math.h>`. The GELU examples call `tanhf` this way. There is no
complete supported-symbol catalog, so a host platform's `libm` is not the
contract: the function must be accepted by MoviTools, resolve from `mlibm.a`,
and produce a kernel ELF that passes the restrictions below.

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

The macro supplies the correct symbol signatures, deterministic unused stubs,
and the tested binary32-to-integer helper. These are NPU3720/MoviTools
link-compatibility definitions, not a general C runtime.

The resulting kernel must be self-contained in `.text`. The current validator
requires `.arg.data` to be empty, so mutable globals, retained writable library
state, unresolved relocations, and general helper runtimes are unsupported.
Read-only constants are usable when the linker places them inside the final
`.text` image.

## Compiler definitions and linker script

`npu.compile(..., definitions=("NAME=123",))` forwards a constrained set of
uppercase decimal definitions. The default NPU3720 linker script is embedded
in the native library. Supplying another script is an advanced override and
does not relax ELF validation.

## Runnable examples

- [FP16 GELU](../examples/example_gelu.py)
- [FP32 GELU](../examples/example_gelu_f32.py)
- [Multi-layer, two-input custom kernel](../examples/example_multilayer_multi_input.py)
- [Mixed-precision graph with FP32 unary and FP16 binary custom branches](../examples/example_mixed_precision_multi_custom.py)

Each example includes a NumPy reference. Successful graph submission is not a
correctness oracle; use equivalent host validation for every new kernel.

For the complete byte-level entry and ELF contract, see
[ACT kernel ELF ABI](../ABI/KERNEL_ELF.md).

[Back to documentation index](README.md)
