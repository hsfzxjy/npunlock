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
| binary custom kernel | static dense FP16 carriers with uniform inputs or a scalar second input; one FP32 scalar-input carrier |
| kernel image | one linked executable image, shareable by selected ranges |
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
  same-precision case with a one-element second input;
- arbitrary input/output aliasing;
- arbitrary tensor arity; or
- automatic layout conversion inside custom code.

The Python serializer knows more IR dtype names than the native execution and
custom-kernel layers support. Serialization capability must not be mistaken
for an execution guarantee.

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
topologically ordered computational nodes and validated positional ACT groups.
One large custom node may instead consume several consecutive groups when a
unique exact-cover mapping is proven: target arity, element width and contract
flags must agree, invocation indices must be consecutive, and their element
counts must sum to the declared output size. It is not a general mapping from
source node names to native ranges.

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
stack use are unsupported.

`mlibm.a` math functions work only when section garbage collection leaves a
self-contained code image and no data dependency.

## Precision

Static dense FP16 is the established general MVP path. FP32 support is limited
to the documented unary carrier and one scalar-second-input binary carrier,
both with precision-conversion suppression and accuracy-mode graph compilation.

Other dtypes, mixed-precision contracts beyond unary FP16/FP32 conversion,
FP32 binary kernels beyond the scalar-second-input carrier, and implicit dtype
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

Shared bindings require complete contiguous FP16 or FP32 arrays allocated by
the same program. Partial views, foreign arrays, mixed copied/shared bindings,
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
