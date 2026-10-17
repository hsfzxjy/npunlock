# NPU3720 ACT kernel ELF: observed ABI

Status: experimental. Last updated 2026-10-17.

This document describes the standalone SHAVE executable produced by
`shavecc`, and the C entry convention used when that code runs as an ACT
kernel inside an NPU3720 graph.

It is an observed compatibility contract, not a published Intel or Movidius
ABI. Read [GRAPH_ELF.md](GRAPH_ELF.md) first for the difference between the
standalone kernel ELF and the native graph ELF that contains it.

## The two ELF files are different

`npunlock` works with two ELF formats:

```text
standalone kernel ELF (ELF32)
  contains the custom SHAVE executable
             |
             | extract validated .text bytes
             v
native graph ELF (ELF64)
  contains scheduling, tensor metadata, and KernelText
```

The graph never receives the complete standalone ELF. `patchblob` extracts
the executable `.text` image and inserts only those bytes into the graph's
`.text.KernelText` section.

## Required standalone ELF structure

The supported linked result has these properties:

| Property | Required observed value |
| --- | --- |
| ELF class | ELF32 |
| byte order | little-endian |
| type | `ET_EXEC` (2) |
| machine | 2 (SPARC value used by these SHAVE tools) |
| entry address | `0x1d000000` |
| executable section | `.text` at `0x1d000000` |
| data section | `.arg.data` at `0x1e000000`, empty |

An executable `PT_LOAD` segment must cover `.text`. Current outputs use
16-byte alignment. The file must contain one unambiguous executable image,
bounded section and program tables, no undefined symbols, and no remaining
relocation sections.

The machine value does not mean the kernel runs on a conventional SPARC CPU.
It is the ELF identity emitted by the MoviTools SHAVE toolchain used here.

## Linker layout

The default linker script places data and code at the addresses expected by
the validated graph carriers:

```ld
. = 0x1e000000;
.arg.data : {
    KEEP(*(.arg.data))
    *(.arg.data)
    . = ALIGN(16);
    *(.data*)
}

. = 0x1d000000;
.text : {
    *(.text*)
    . = ALIGN(16);
    *(.gnu.linkonce.text.*)
    . = ALIGN(16);
    *(.rodata*)
    . = ALIGN(16);
    KEEP(*(.uuid.rodata*))
    *(.uuid.rodata*)
}
```

Read-only constants that the linker places in the output `.text` section are
part of the extracted code image. Linker padding inside that section is also
preserved.

The current contract requires `.arg.data` to be empty. Nonempty mutable data
would need graph placement and relocation behavior that has not been
validated. It must not be approximated by concatenating ELF sections.

The Apache-2.0 default script is stored at
[`src/shavecc/shave_kernel.ld`](../src/shavecc/shave_kernel.ld) and embedded in
the library at build time. A caller may explicitly supply a replacement, but
the returned ELF must still pass the same validation.

## Build pipeline

`shavecc` runs three in-memory stages:

```text
C source bytes
  -> moviCompile64.dll
SHAVE assembly bytes
  -> moviAsm64.dll
relocatable ELF32 object
  -> moviLLD64.dll + linker script
linked executable ELF32
```

The confirmed target selector is `3720xx`. The effective arguments are:

```text
moviCompile.dll -cc1 -triple shave -target-cpu 3720xx
  -S -O2 -x c entry.c -ffunction-sections -fdata-sections -o -

moviAsm.dll --cv 3720xx --noSPrefixing

moviLLD.dll -flavor gnu -EL -e <entry> -z max-page-size=0x10
  --gc-sections <MoviTools-root>\lib\mlibm.a
```

The source, assembly, object, linker script, and result stay in memory. The
nominal `entry.c` name is a compiler argument, not an intermediate file.

`mlibm.a` supplies math functions such as `tanhf`. Section garbage collection
is important: without `--gc-sections`, unrelated archive members can introduce
writable global data in `.arg.data`, making the kernel unsupported. With
garbage collection, the validated GELU examples retain only the reachable math
closure and still produce an empty data section.

Passing these structural checks does not prove that an arbitrarily large math
closure will execute. In one NPU3720 probe, a runtime-selected four-function
FP32 image of 3,056 bytes executed, while adding a fifth function produced a
3,504-byte image that passed validation but caused an NPU device error. The
original carrier extent was 5,280 bytes in both cases, so fitting within that
extent is not a sufficient execution criterion. This is an observed boundary,
not a universal byte threshold.

Simple arithmetic kernels do not require a runtime archive beyond this
linker's normal inputs.

## C entry point

The execution-tested source signature is:

```c
void controlled_act(unsigned layerParams);
```

The compiler passes `layerParams` in integer register `i18`; ordinary compiled
returns use `i30`. This describes the entry behavior needed by the current
carriers, not a complete SHAVE calling-convention specification.

`layerParams` is the low 32-bit address of the parameter block selected for
the current graph invocation. Successful kernels read pointer-bearing fields
with explicit little-endian 32-bit loads. Runtime self-check kernels observed
zero in the upper 32-bit word of both input and output data addresses for the
host-visible FP16 and FP32 unary paths. A nonzero upper word remains
unsupported, and the 32-bit entry argument cannot report a hypothetical upper
half of `layerParams`.

Kernel source may start with the virtual include
`#include <npunlock/npu3720_kernel.h>`. `shavecc` expands it from an embedded
copy before compilation. Its target-scoped helpers implement the descriptor
loads and guarded element-count calculation described below; they are a
convenience over this observed contract, not evidence of a broader ABI.
`act_abi_dimension()` returns zero for an invalid axis, and invocation loading
rejects a dimension product that would overflow its 32-bit element count.

## Reading tensor descriptors

The parameter block starts with one or more 0x28-byte tensor descriptors. A
kernel normally needs these fields:

```text
descriptor +0x00  low 32 bits of tensor data address
descriptor +0x04  observed-zero upper 32 bits of tensor data address
descriptor +0x0c  rank as u32
descriptor +0x10  low 32 bits of dimensions-array address
```

The dimensions array contains `rank` little-endian 32-bit values. A custom
kernel should:

1. reject rank zero or a rank larger than its own bound;
2. reject zero dimensions;
3. check multiplication before calculating the element count; and
4. never process beyond the element count described for this invocation.

One execution probe compiled from a logical FP16 graph tensor shaped `(4,8)`
produced four unary ACT invocations. Every entry reported rank 4, dimensions
`(8,1,1,1)`, and element count 8. A kernel that validated those fields and
derived its reversal width from dimension 0 matched the graph's four rows
exactly. This is one compiler/carrier observation, not a general mapping from
logical graph axes to invocation axes.

Host-side `patchblob` validation checks the fields that the small kernel does
not: static shape, dense bit strides, supported precision, CMX placement,
matching input/output spans, and non-aliasing output.

Validated descriptor positions are:

```text
unary kernel:
  input  at layerParams + 0x00
  output at layerParams + 0x28

binary kernel:
  input A at layerParams + 0x00
  input B at layerParams + 0x28
  output  at layerParams + 0x50

validated five-input kernel:
  inputs 0..4 at layerParams + index * 0x28
  output      at layerParams + 0xc8
```

Both internal operands and two independently bound host inputs have been
validated for a narrow static dense FP16 binary carrier. A separate
experimental FP16/FP32 carrier presented eight-element data and output chunks
beside a one-element scalar input; reading `input_b[0]` for every output
matched the host oracle. A separate `[4,8]` plus `[1,8]` carrier presented the
row bias as a dense eight-element operand to each invocation; this establishes
the descriptor view, not physical copying of the bias.

`ACT_ABI_LOAD_INVOCATION32_OR_RETURN` derives rank, dimensions, and
`element_count` from tensor record zero. That is sufficient when record zero
has the output extent. For scalar-first or future unequal-count contracts, use
`ACT_ABI_LOAD_INVOCATION32_FROM_RECORD_OR_RETURN` with an explicitly reviewed
record, or use `ACT_ABI_LOAD_OUTPUT_INVOCATION32_OR_RETURN(layerParams,
input_count, invocation)` to derive the loop extent from the output record.
Both helpers keep `invocation.params32` anchored at the original parameter
block, so input/output pointer macros retain their normal record indices.

The output-record helper was execution-tested on FP16, FP32, U16, and U32
`(1,8)->8` scalar-first carriers. Loading record zero instead produced a count
of one and left most output elements unwritten; kernel writers must select the
record whose shape controls their loop.

One `FakeQuantize` carrier exposed five independently bound, equal-sized FP16
input records followed by one output record. Four invocations each advertised
`(8,8,8,8,8) -> 8`, and a custom kernel that distinguished all operands
matched its host oracle. This establishes one five-input layout, not arbitrary
input counts or general `FakeQuantize` semantics.

## Invocation-local execution

The graph scheduler divides a tensor into invocation-local chunks. A custom
kernel receives only the descriptor for its current chunk; it does not
automatically receive a full graph-global tensor view.

Confirmed FP16 kernels include:

- elementwise add-one over chunks of 8 and 16 elements;
- one code image selected by several graph ranges;
- FP32 intermediate arithmetic with FP16 output;
- two-input arithmetic with independently bound host inputs;
- five-input arithmetic with independently bound host inputs;
- a fused five-input residual/RMS/affine/gate pipeline using `sqrtf`;
- a distinct same-file non-inlined helper with a data-dependent branch;
- a source-level 128-byte volatile local scratch array;
- three nested same-file helpers, a local context structure, and a five-tap
  loop using a retained 20-byte coefficient object;
- a source-level 256-byte volatile local scratch array;
- three-pass invocation-local maximum/sum reductions around `mlibm.a`'s
  `expf` implementation;
- insertion sort with a data-dependent inner loop, indirect local indexing,
  and invocation-local output permutation;
- rank/dimension validation and dimension-derived indexing for one observed
  `(8,1,1,1)` invocation layout;
- runtime selection between two same-file helpers through one bounded local
  function pointer and dispatcher;
- runtime indexing of a three-entry function table initialized explicitly in
  invocation-local storage;
- invocation-local chunks of 64, 128, and 2048 elements; and
- reads across the full advertised 4096-byte span of a 2048-element chunk.

The last result is a lower bound for local reach, not permission to cross an
invocation boundary. Neighbor-based kernels must define their boundary
behavior within each chunk.

A structured kernel with a distinct 50-byte local helper symbol, branching,
invocation-local neighbor reads, and `volatile float scratch[32]` has executed
successfully. Its final ELF had no relocations or undefined symbols. This
establishes one same-file call and bounded source-level local storage;
recursion, the general callee-save convention, and arbitrary runtime-library
behavior remain outside the supported contract.

A later minimal probe isolated local storage from code-image size. It wrote
and hashed every byte of a volatile byte array while the executable image
remained 416 bytes for every tested array size. A 7,200-byte array returned
the exact 32-element FP16 result in an early run, but later runs silently
corrupted half the output or returned Level Zero device result `0x70000001`.
A 7,168-byte body succeeded in four observed runs. At 7,184 bytes, one of five
runs was exact and four silently corrupted the second 16-element half. An
8,192-byte array returned the device error. These sizes expose an unstable
stack/frame boundary; they are not a general capacity promise because frame
overhead, calls, compiler output, prior execution state, and carrier scheduling
may change it. The retained ELF had no `.debug_frame`, `.eh_frame`, or target
stack-size note from which `patchblob` could enforce a frame bound.

A later kernel retained two local transform symbols and a separate dispatcher.
It selected a same-file target through a volatile local function pointer and
executed both choices correctly. Its final ELF had no relocations or undefined
symbols. This validates one bounded indirect-call shape, not general function
pointer storage, external call targets, callbacks, or arbitrary call graphs.

A follow-up retained three transform symbols, a table selector, and a separate
dispatcher. An explicitly assigned `volatile` invocation-local table supported
runtime indexing and exact execution through copied and shared buffers. The
1,468-byte ELF contained a 688-byte executable `.text`, empty `.arg.data`, and
no relocations or undefined symbols. In contrast, file-scope and locally
initialized constant pointer tables both produced nonempty `.arg.data` with
the tested toolchain and were rejected. This validates the invocation-local
runtime-initialization pattern, not static or global pointer-table storage.

The nested-window kernel produced three local helper symbols of 260, 70, and
35 bytes plus a 20-byte global `window_weights` object in `.text`. MoviTools
marked the combined `.text` section writable and executable even though the
source object was `const volatile`. The final image still had an empty
`.arg.data` and no relocations or undefined symbols. This is evidence for an
embedded constant table, not for a generally separate read-only data segment.

An invocation-local softmax-like kernel linked `expf` and `floorf`, performed
two FP32 scalar reductions, reused a source-level 256-byte scratch array, and
matched a chunk-aware host oracle. Its ELF file was 53,796 bytes because it
retained non-loadable debug sections, while the single executable `.text`
segment installed into the graph was 976 bytes. The file size is therefore not
the injected code-image size.

An invocation-local insertion-sort kernel produced a 608-byte executable
`.text` image. It used a source-level 256-byte FP32 scratch array and a
data-dependent `while` loop to shift indirectly indexed values. Copied and
shared execution exactly matched chunk-wise host sorting. This proves bounded
local permutation, not cross-invocation movement or general dynamic storage.

A fused five-input kernel combined residual addition, an FP32 sum-of-squares
reduction, `sqrtf`, per-element scale and bias, and gating. The 53,640-byte ELF
contained an 864-byte executable `.text` image and no relocations or undefined
symbols. All five eight-element input records affected each eight-element
output chunk. This validates composition on the observed carrier, not general
RMSNorm lowering or arbitrary high-arity layouts.

## How the code image is installed

After kernel validation, `patchblob` performs this mapping:

```text
kernel ELF .text bytes
  -> aligned append inside graph .text.KernelText
  -> selected ActKernelRange relocation points to the append offset
  -> selected ActKernelRange extent records the image size
```

The standalone virtual address and supported graph range both use
`0x1d000000`. Several ranges may reference one appended image. The current
patcher uses 0x400-byte append alignment and 0x80 bytes of tail padding as
conservative, execution-tested values; they are not proven minima.

## Validation performed by `shavecc`

Before returning success, the library checks at least:

- ELF magic, class, byte order, type, machine, and entry address;
- section, program, string, and symbol table bounds;
- a nonempty executable `.text` at the expected address;
- executable-load coverage of `.text`;
- `.arg.data` at the expected address with size zero;
- absence of undefined symbols;
- absence of remaining relocation sections; and
- overflow safety for every derived offset and extent.

These checks establish structural compatibility, not computation correctness.
The selected graph carrier must also pass its tensor-contract checks, and an
application must compare output with a host oracle when validating new kernel
semantics.

## Unsupported assumptions

Do not infer support for:

- nonempty `.arg.data` or mutable global/static storage;
- arbitrary helper libraries or data-bearing library closures;
- unresolved, dynamic, or runtime relocations;
- multiple executable images or alternate entry addresses;
- nonzero high tensor-address halves or a parameter block outside the observed
  32-bit entry-address convention;
- dynamic shapes, non-dense layouts, or arbitrary precisions;
- general broadcasting or unequal binary shapes beyond the tested scalar and
  descriptor-level row-broadcast cases;
- arbitrary arity beyond the validated unary, binary, and five-input layouts;
- graph-global neighborhoods or cross-chunk halos;
- large stacks, exceptions, or a C++ runtime; or
- SHAVE targets other than `3720xx`.

## Related public documentation

- [GRAPH_ELF.md](GRAPH_ELF.md) describes the native graph container and range
  records.
- [MOVITOOLS.md](MOVITOOLS.md) describes the compiler DLL boundary.
- [C API](../docs/C_API.md) documents `shavecc` and `patchblob` ownership and
  error handling.
