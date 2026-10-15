# Current limitations

[Documentation index](README.md)

`npunlock` exposes an experimental path that has been validated on a narrow
hardware and graph configuration. The project fails closed when compiler
output does not match that contract.

## Supported baseline

| Area | Confirmed scope |
| --- | --- |
| host | Windows x64 |
| hardware | Meteor Lake / NPU3720 (`0x7d1d`) |
| SHAVE target | `3720xx` |
| graph compiler | current path validated with version 8.3 |
| primary custom tensors | static, dense FP16 |
| FP32 | validated unary accuracy-mode carrier, one same-precision scalar-input binary carrier, and explicit unary FP16/FP32 conversion |
| I32 | one same-precision unary `Abs` carrier with 16-element invocation descriptors |
| I8 | one same-precision unary `Abs` carrier with 16-element invocation descriptors |
| binary custom kernel | static dense FP16 carriers with uniform inputs, a scalar second input, or the validated descriptor-level `(4,8)` plus `(1,8)` row-broadcast carrier; one FP32 scalar-input carrier |
| higher custom arity | one static dense FP16 five-input `FakeQuantize` carrier with equal per-invocation spans |
| kernel image | one linked executable image, shareable by selected ranges |
| ordinary graph I/O | static dense FP16, FP32, U8, I8, U16, I16, U32, and I32 through copied or shared buffers |
| restricted graph input | I64 only as caller-provided indices for the validated U8 Gather graph |
| execution | installed Intel NPU driver through Level Zero |

No other Intel NPU generation is claimed to work merely because it exposes a
similar driver interface.

## Static shapes and layouts

Symbolic shapes contain only positive fixed dimensions. Dynamic dimensions
are unsupported. Custom ACT tensors must match the validated dense descriptor
contract.

There is no general support for:

- arbitrary strides or layouts;
- broadcasting or unequal binary input shapes beyond the implemented static
  one-element second input and one descriptor-level FP16 `(4,8)` plus `(1,8)`
  row broadcast;
- arbitrary input/output aliasing;
- arbitrary tensor arity beyond the validated unary, binary, and five-input
  layouts; or
- automatic layout conversion inside custom code.

The Python serializer knows more IR dtype names than native execution supports.
Ordinary graph execution is validated for FP16, FP32, U8, I8, U16, I16, U32,
and I32, while I64 is limited to the Gather-index role. Custom ACT-kernel
contracts remain limited to the FP16/FP32 cases below and one unary carrier
each for I32 and I8. Serialization capability must not be mistaken for either
execution or custom-kernel support.

## Invocation-local memory

Custom kernels operate on compiler-defined invocation chunks. They do not
receive a graph-global tensor address or global element index. Reads have been
validated only within the span advertised by the current invocation.

Cross-chunk halos, graph-global neighborhoods, and access beyond descriptor
bounds are unsupported.

## Carrier and mapping constraints

Not every graph operation produces an ACT kernel. The compiler may place an
operation on the DPU, fuse it, optimize it away, or insert conversion groups.
Such changes can break a presumed source-to-ACT relationship.

Automatic Python selection normally requires a one-to-one match between all
topologically ordered computational nodes and validated positional ACT groups,
with matching input arity and tensor precision. This discovery contract is
validated only for graph compiler 8.3 and NPU3720. `patchblob` requires those
values, a one-or-two-tile count, loader-ELF 1.2.0, and mapped-inference 7.0.4
in the native blob's vendor compatibility metadata and version notes. The
platform-info record must independently identify the NPU3720 architecture.
Python also checks the graph compiler version reported by the driver for
graphs it compiles. Missing or different blob provenance is rejected,
including for raw explicit targets, because the record parser itself is
compiler-specific.
One large custom node may instead consume several consecutive groups when a
unique exact-cover mapping is proven: target arity, element width and contract
flags must agree, invocation indices must be consecutive, and their element
counts must sum to the declared output size. It is not a general mapping from
source node names to native ranges.

An `ActGroup`'s aggregate element count is descriptor coverage, not a logical
graph tensor size. One logical 32-element FP16 carrier exposed four 16-element
invocations: two partitions duplicated across two tiles. Summing them gives 64,
while treating every equal count as a replica would give 16. The current
metadata cannot distinguish those cases without schedule/DMA routing analysis,
so single-group positional mapping does not claim an element-count proof.

Advanced explicit invocation/range targets remain available only for layouts
the caller has independently validated. DPU-only graphs have no patchable ACT
carrier.

## Kernel ELF restrictions

The linked custom kernel must have:

- one executable `.text` image at the validated address;
- an empty `.arg.data` section;
- no undefined symbols;
- no remaining relocations; and
- the expected entry symbol and target.

Mutable globals, unresolved fixups, nonempty kernel data, multiple load
images, arbitrary helper runtimes, C++ exceptions, and large or recursive
stack use are unsupported. A bounded same-file helper call, branching, and a
source-level 128-byte local scratch array are execution-tested; this does not
establish a general calling convention or stack-size limit.

The NPU3720 kernel header deliberately uses a 32-bit `layerParams` argument and
low-32-bit pointer loads. Runtime probes observed zero upper words for input
and output data addresses on the host-visible FP16 and FP32 unary paths.
Nonzero upper data addresses, a wider parameter-block address, hidden replica
routes, and other NPU generations remain unsupported.

One five-tap kernel also validates a three-helper nested call chain, a local
context structure, a source-level 256-byte scratch array, and a retained
20-byte `const volatile` coefficient object. The linker placed that object in
a writable/executable `.text` section, not a separately protected read-only
segment. Recursion and arbitrary call graphs remain unsupported.

One minimal probe repeatedly wrote and read a 7,168-byte volatile local array,
but 7,184-byte runs alternated between correct output and silent half-output
corruption. A 7,200-byte body also produced corruption and a device error, and
8,192 bytes caused a device error. This is not a stable stack-capacity
contract: frame overhead, kernel structure, and scheduling also matter. Keep
production scratch conservative and compare complete outputs with a host
oracle.

Maximum and sum reductions plus `expf` have been validated only inside one ACT
invocation. A logical tensor may span several invocations, so this supports
chunk-local normalization but does not establish graph-global reduction,
cross-invocation communication, or general axis semantics.

Data-dependent loops and indirect scratch indexing are validated by one
insertion-sort kernel, again within each invocation independently. There is no
graph-global sort contract, cross-invocation permutation, index output, dynamic
allocation, or guarantee for unbounded loop runtimes.

Invocation dimensions are compiler-local metadata, not the original graph
shape. One logical `(4,8)` tensor was observed as four rank-4 `(8,1,1,1)`
entries. Other shapes, carriers, compiler versions, or axis mappings must be
inspected and validated independently.

Function pointers are validated for bounded same-file dispatch, including one
runtime-initialized invocation-local table of three targets. Global or
statically initialized pointer tables require unsupported kernel data in the
tested toolchain. Targets from tensor data, external functions, callbacks,
recursion, and arbitrary indirect call graphs remain unsupported.

One five-input carrier runs a fused residual/RMS/affine/gate kernel with
`sqrtf`, but its reduction is local to each eight-element invocation. It does
not establish standard graph-global RMSNorm semantics, arbitrary five-input
operators, or other high-arity layouts.

`mlibm.a` math functions work only when section garbage collection leaves a
self-contained code image and no data dependency. Eight FP32 unary functions
are execution-tested individually, but combining several retained math paths
can still fault even when the linked image is smaller than the carrier's old
code extent. There is not yet a general executable-composition or code-size
limit; validate the complete combined kernel on the target NPU.

Native graph parsing accepts only the observed compiler-8.3/NPU3720 ELF
container and section-table form. Required sections must retain their observed
metadata, file-backed sections may not overlap, and their file offsets must
satisfy the declared power-of-two alignments. Older `npunlock`-generated blobs
that shifted sections without preserving those alignments can still be raw
driver artifacts, but are not accepted for further discovery or patching.
ACT relocation sources are likewise restricted to the observed
`.symtab.tasks`/`.symstrtab` section symbols and VPU runtime-table index
`0xff20`. Tensor data pointers must use runtime symbol 0. Other symbol-table
layouts and loader namespaces are not inferred from similar-looking records.
Only the observed ACT relocation types 3, 4, and 5 are accepted. Their eight-
or four-byte target writes must be naturally aligned and contained within the
target section. Custom-kernel pointers use type 4 (`R_VPU_32`); wider device
addresses are not established by the ELF64 relocation record.

## Precision

Static dense FP16 is the established general MVP path. FP32 support is limited
to the documented unary carrier and one scalar-second-input binary carrier,
both with precision-conversion suppression and accuracy-mode graph compilation.

I32 ordinary graph inputs and outputs are validated for Abs, Concat,
Transpose, and FP32-to-I32 Convert graphs. Custom execution is additionally
validated for one static `[1,32]` same-precision unary `Abs` carrier: four
invocations each described 16 signed 32-bit elements, and a custom affine
kernel matched all 32 outputs. Binary I32 carriers, mixed integer contracts,
other shapes, and integer-overflow semantics remain unsupported.

U8 has been validated only for ordinary graph inputs and outputs, including a
constant-pattern Reshape, U8-to-FP32 Convert, and Gather with I32 indices. It
also covers one static unequal VariadicSplit with three outputs and one static
multi-axis positive-step Slice. Constant-mode Pad is validated with nonnegative
static pads and a scalar U8 fill value. U8 is not a supported custom
ACT-kernel precision.

I8 ordinary graph inputs and outputs are validated for positive-axis
Squeeze/Unsqueeze and I8-to-FP32 Convert. Graph compiler 8.3 rejected one
equivalent negative-axis form, so negative axes are not in the verified
execution contract. Custom execution is additionally validated for one static
`[1,32]` same-precision unary `Abs` carrier: four invocations each described
16 signed bytes, and a bounded affine kernel matched all 32 outputs. Binary I8,
U8 custom kernels, saturation, quantization semantics, and other shapes remain
unsupported.

I16 has been validated for ordinary graph input/output through an equal static
three-way Split with a scalar constant axis. Graph compiler 8.3 rejected the
tested I16-to-FP32 Convert, so no I16 conversion path is claimed. I16 is not a
supported custom ACT-kernel precision.

U16 has been validated for ordinary graph input/output through an equal static
two-way Split. The tested U16-data Gather compiled and executed without a
reported device error but returned incorrect values, so U16 Gather is not in
the supported contract. U16 is not a supported custom ACT-kernel precision.

U32 has been validated for ordinary graph input/output through an equal static
three-way Split, including exact values through `2**32 - 1`. This does not
establish U32 arithmetic operators or a U32 custom ACT-kernel precision.

I64 has been validated only as a caller-provided index input to the tested U8
Gather graph. An I64 Split compiled and ran, but values outside signed 32-bit
range were silently narrowed and one extreme shared run returned a device
error. General I64 graph data and outputs are therefore unsupported.

Other dtypes, mixed-precision custom contracts beyond unary FP16/FP32
conversion, FP32 binary kernels beyond the scalar-second-input carrier, I32
or I8 custom kernels beyond the unary carriers above, and implicit dtype
conversion are unsupported.

One graph containing independent FP32-unary and FP16-binary custom branches
has been executed successfully with explicit target selection. Each of those
ACT groups remains internally single-precision.

The version 2 patch contract describes every input and output record
independently. It supports the observed unary FP32-to-FP16 and FP16-to-FP32
layouts when input and output have equal element counts and their spans match
their precision. Both compiler-generated tile replicas must be selected. The
Python frontend exposes these groups but requires explicit prepared-graph
binding; it does not treat the replicas as partitions or automatically map an
inserted conversion to a source node. The connected conversion example has
executed both directions in one graph and matched its host oracle exactly.

A later `[1,32]` binary-carrier experiment preserved a host-provided `[1,1]`
operand as one scalar ACT descriptor beside eight-element input/output chunks.
Custom FP16 and FP32 kernels read that scalar and matched two 32-element host
oracles exactly. Discovery and patching now support that exact contract through
`PATCHBLOB_CONTRACT_INPUT_1_SCALAR`. Other unequal shapes, scalar positions,
broadcast axes, and per-record contracts remain unsupported.

## Software dependencies

The project does not depend on OpenVINO as a Python package, runtime, or graph
compiler frontend. It does serialize OpenVINO-format IR because the Intel NPU
driver accepts that representation.

Custom C compilation currently requires caller-supplied proprietary MoviTools.
`npunlock` does not redistribute or automatically download them. The current
installed Intel NPU driver remains required for graph compilation and hardware
execution.

## Correctness and stability

Inference runs in-process for both copied and shared tensors. Fence waits are
finite, but a driver API call that never returns cannot be killed independently
of the application. Shared arrays remain opt-in; ordinary inputs and outputs
are copied through internal Level Zero host allocations.

Shared bindings require complete contiguous arrays allocated by the same
program, using one of the ordinary graph dtypes above or I64 for a validated
Gather index input. Partial views, foreign arrays, mixed copied/shared bindings,
and resizing shared storage are rejected.

Driver acceptance, graph creation, or successful command submission does not
prove that a new custom kernel is semantically correct. Validate output against
a host oracle.

The observed binary formats and DLL entry points are not published stable
vendor ABIs. Changes to MoviTools, the Intel graph compiler, or the native graph
format may require new validation.

Possible Linux and newer-generation paths are documented separately as
unverified hypotheses in [Porting to Linux and newer NPUs](PORTING.md). They do
not expand the supported baseline above.

For implementation details, see [How npunlock works](HOW_NPUNLOCK_WORKS.md)
and the [binary-format references](README.md#binary-format-references).

[Back to documentation index](README.md)
