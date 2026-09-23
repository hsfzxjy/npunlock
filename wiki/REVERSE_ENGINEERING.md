# Reverse-engineering breakthroughs

[Documentation index](README.md)

`npunlock` exists because Intel's public NPU stack exposes graph compilation,
not a user-facing route from C source to a custom ACT-SHAVE operation. The
project's central breakthrough was discovering that these two jobs do not have
to be solved together:

1. Intel's graph compiler can still create tensor placement, scheduling, data
   movement, barriers, and an ACT invocation environment for a known operator.
2. A separately compiled SHAVE code image can replace that operator's ACT
   implementation while the surrounding execution carrier remains intact.

That observation turned an undocumented native graph into a practical,
testable custom-kernel path. This page records the important milestones behind
that result without treating every observed field as a general vendor ABI.

Status: experimental. Last updated 2026-09-23.

## How the conclusions were established

The work used differential compilation and controlled mutation rather than
assigning meanings to bytes from appearance alone:

- compile closely matched graphs that differ in one operation, shape, or tile
  setting;
- compare their native ELF sections and relocation records;
- change one candidate field or component at a time;
- execute both the original and modified graph with discriminating inputs;
- compare every output against an independent host calculation; and
- retain negative results when a plausible contract failed.

On this page, **confirmed** means that byte-level evidence and a controlled NPU
execution agree. **Inferred** means that several observations support an
interpretation, but no experiment has isolated it completely. **Open** means
that the current project deliberately makes no compatibility claim.

## 1. ACT code is selected independently from its invocation

The first useful native-graph discovery was that an ACT invocation selects an
`ActKernelRange`, and that range selects a slice of the graph's
`.text.KernelText` section. Closely matched graphs containing different ACT
operations exposed distinct code slices while retaining comparable invocation
and scheduling structures.

A causal test exchanged the selected ranges for two operations while leaving
their tensor parameters and scheduling untouched. The output operations
exchanged accordingly. Changing only the invocation range indices produced
the same semantic swap.

**Confirmed:** for the tested NPU3720 compiler family, code selection can be
separated from the invocation's data and execution environment. A code address
alone is not a safe logical-operation identifier, because several invocations
or operations may share an implementation.

See [the native graph ELF ABI](../ABI/GRAPH_ELF.md) for the observed record and
relocation layout.

## 2. An ACT implementation can move between compatible graphs

The next experiment used two compiler-generated graphs whose execution
carriers were byte-compatible but whose ACT operations differed. Copying the
source implementation's `KernelText` into the target and changing the selected
range extent reproduced the source operation's output exactly. Parameters,
DMA tasks, barriers, DPU tasks, and invocation records did not need to change.

This established the carrier idea: the graph compiler supplies a valid runtime
environment, while the selected ACT implementation supplies the computation.

**Confirmed:** code-only substitution works when the source and target share a
compatible tensor and invocation contract. It does not imply that code can be
moved between arbitrary graphs.

## 3. The graph contains a linked SHAVE image, not a nested kernel ELF

Standalone NPU3720 ACT ELFs were compared with their compiler-generated graph
counterparts. Their executable `.text` bytes matched the corresponding graph
code slices exactly. The graph stored the linked virtual address and image
extent, but not the complete standalone ELF container.

A follow-up test appended a standalone image to `.text.KernelText`, grew the
section, rebased later ELF file offsets, and retargeted one ACT range through
its relocation. The modified graph loaded and produced the expected output.

**Confirmed:** the practical installation unit is validated executable
`.text`. The minimum working graph mutation is:

```text
append aligned code image
rebase affected ELF file offsets
update selected range extent
update selected KernelText relocation
```

Other compiler-generated execution structures remain unchanged. The current
alignment and padding choices are conservative tested values, not proven
hardware minima.

## 4. The proprietary MoviTools DLLs can form an in-memory toolchain

The available MoviTools package did not expose a documented SDK, but its
compiler, assembler, and linker DLLs exported entry points that behave like
in-memory command-line tools. Controlled calls established a complete pipeline:

```text
C source bytes
  -> SHAVE assembly
  -> relocatable ELF32 object
  -> linked ELF32 executable
```

The output is a little-endian SPARC ELF32 image linked for the `3720xx` target,
with executable code at `0x1d000000`. The supported custom-kernel contract
requires empty `.arg.data`, no undefined symbols, and no unresolved
relocations.

**Confirmed:** caller-provided C can be compiled entirely through memory
buffers into a standalone image suitable for graph installation. Because the
DLL entry points may fault, exit, or hang like command-line programs,
`npunlock` invokes them in bounded worker processes and copies all results
across the process boundary.

See [the MoviTools contract](../ABI/MOVITOOLS.md) and
[standalone kernel ELF ABI](../ABI/KERNEL_ELF.md) for the version-specific
details.

## 5. A C function can obey the ACT invocation contract

With code generation and graph installation established separately, the next
step was to connect them. A C entry point named
`controlled_act(unsigned layerParams)` received a usable parameter-block
address. For the tested unary carrier, consecutive tensor descriptors supplied
the input pointer, output pointer, rank, dimensions, and element count.

The first custom kernel wrote a marker to only one invocation. A second kernel
computed FP16 add-one. Replacing one range changed only its assigned output
chunk; replacing all ranges produced the intended result for every element.
Both results matched independent raw-FP16 host calculations.

**Confirmed:** ordinary compiled C control flow, loads, stores, loops, return,
and FP32 intermediate arithmetic can execute as an NPU3720 ACT kernel when the
function follows the observed carrier contract. A valid ELF by itself is not
enough: an earlier apparently valid dummy image loaded but caused device loss,
demonstrating that packaging alone does not establish runtime compatibility.

The bundled [`npu3720_kernel.h`](../include/npunlock/npu3720_kernel.h) exposes
only the helpers required by this tested convention.

## 6. One code image can serve several work partitions

The same compiled kernel was then tested with different graph shapes, CMX
placements, and tile settings. It followed relocated tensor pointers and
per-invocation dimensions instead of relying on fixed addresses or a fixed
loop count. Multiple ranges were also retargeted to one shared appended code
image and produced correct results.

**Confirmed:** one custom image can serve multiple compatible invocation
chunks. A request for one NPU tile does not necessarily produce one ACT
invocation; the compiler remains responsible for work partitioning.

This is why kernel code must treat the descriptor's element count as
invocation-local. It must not assume that it sees the entire graph tensor.

## 7. Local spans and two-input kernels are usable

Increasingly large unary carriers established successful reads across an
advertised local input span of 2048 FP16 elements, or 4096 bytes. A nonlinear
neighbor kernel matched a host reference while respecting each invocation's
local endpoints.

Separate binary experiments established the observed descriptor order for two
inputs and one output. A later graph bound two independent host inputs and ran
a custom weighted mix successfully.

**Confirmed:** the supported static dense FP16 contract includes unary and
two-input kernels, local indexing within the advertised invocation span, and
independently bound graph inputs.

**Open:** cross-invocation reads, halo exchange, graph-global coordinates,
broadcasting, unequal input shapes, and the maximum local span. The 4096-byte
result is a tested lower bound, not a limit or a general memory-access promise.

## 8. Graphs can be compiled without loading OpenVINO

The installed Intel driver advertises an `NGRAPH_LITE` input accepted through
the Level Zero graph extension. Reconstructing its small in-memory envelope
made it possible to send OpenVINO-format IR XML and weights directly to the
driver, request graph compilation, and retrieve the complete native binary.

Static graphs with and without weight buffers compiled and executed. The
exported weightless graph also survived a save/reload round trip. The confirmed
path uses `pfnCreate2`; an attempted `pfnCreate3` call terminated and is not
part of the supported implementation.

**Confirmed:** `npunlock` does not need the OpenVINO runtime, Python package, or
NPU plugin. The serialization is still OpenVINO-format IR, and Intel's
installed graph compiler still decides how that graph is lowered.

## 9. A narrow FP32 path can avoid hidden FP16 conversion

A plain FP32 activation carrier was observed to lower into three ACT groups:
FP32-to-FP16 conversion, the activation, and FP16-to-FP32 conversion. Replacing
only the middle operation therefore preserved the graph's hidden precision
loss.

Adding the IR runtime attribute that disables FP16 precision conversion and
requesting accuracy execution mode instead produced one FP32 ACT group. A
custom FP32 GELU kernel then matched its host reference with maximum absolute
error `2.38419e-07`.

**Confirmed:** this configuration provides a precision-preserved unary FP32
carrier on the tested driver and NPU3720. It is not evidence for arbitrary
FP32 graphs, FP32 binary kernels, or mixed-dtype ACT invocations.

## 10. Independent mixed-precision custom branches can share one graph

On 2026-09-23, a single graph executed two independent custom branches: one
unary FP32 operation and one two-input FP16 operation. Both branches matched
their host references exactly.

The Intel compiler emitted the independent ACT groups in a different order
from the symbolic graph's output traversal. Automatic positional mapping
therefore rejected the graph correctly. A preflight compilation identified
the unique groups by input arity and element width, after which explicit
validated targets installed both kernels.

**Confirmed:** One native graph can execute independent FP32-unary and
FP16-binary custom branches; explicit ACT-group preflight handles the
compiler's branch reordering.

A connected FP32-to-FP16 experiment also reached graph compilation, but the
current discovery validator rejected the ordinary conversion group because
its input and output byte spans differ. This result does not establish a
connected mixed-precision custom pipeline or mixed-dtype ACT invocation.

The complete runnable case is
[`example_mixed_precision_multi_custom.py`](../examples/example_mixed_precision_multi_custom.py).

## What remains deliberately unresolved

The experiments above reconstruct a useful path, not a complete Intel NPU SDK.
The project does not currently claim support for:

- dynamic shapes, arbitrary strides, layouts, or precisions;
- cross-invocation or graph-global memory access;
- nonempty custom kernel data sections or unresolved code/data fixups;
- a general mapping from source node names to native ACT records;
- arbitrary graph/compiler versions or NPU generations; or
- an official or stable vendor ABI for MoviTools or native graph records.

The implementation fails closed when a graph, kernel, or mapping falls outside
the observed contracts. See [Current limitations](LIMITATIONS.md) for the
product-facing compatibility boundary and [How npunlock works](HOW_NPUNLOCK_WORKS.md)
for the resulting end-to-end design.

[Back to documentation index](README.md)
