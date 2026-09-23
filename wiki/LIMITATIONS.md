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
| FP32 | validated unary accuracy-mode carrier only |
| binary custom kernel | one static dense FP16 carrier family, including independent host inputs |
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
- broadcasting;
- unequal binary input shapes;
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
to the documented unary carrier with precision-conversion suppression and
accuracy-mode graph compilation.

Other dtypes, mixed-precision custom groups, FP32 binary kernels, and implicit
dtype conversion are unsupported.

One graph containing independent FP32-unary and FP16-binary custom branches
has been executed successfully with explicit target selection. Each ACT group
remains internally single-precision. A connected FP32-to-FP16 experiment
compiled through the Intel graph compiler, but target discovery rejected the
ordinary conversion group because its input and output spans differ. This does
not establish connected mixed-precision custom pipelines or mixed-dtype ACT
invocation contracts.

## Software dependencies

The project does not depend on OpenVINO as a Python package, runtime, or graph
compiler frontend. It does serialize OpenVINO-format IR because the Intel NPU
driver accepts that representation.

Custom C compilation currently requires caller-supplied proprietary MoviTools.
`npunlock` does not redistribute or automatically download them. The current
installed Intel NPU driver remains required for graph compilation and hardware
execution.

## Correctness and stability

The default inference path is isolated in a finite-lived worker process.
Host/NPU shared arrays require an opt-in in-process graph session because the
validated NPU driver does not export its host-visible Level Zero allocations
between processes. Fence waits remain finite, but a driver API call that never
returns cannot be killed independently of the application in shared mode.

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
