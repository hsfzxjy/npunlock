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

Status: experimental. Last updated 2026-09-27.

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
general broadcasting and unequal input shapes beyond the scalar case in
section 14, and the maximum local span. The 4096-byte result is a tested lower
bound, not a limit or a general memory-access promise.

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

## 11. Mixed-precision ACT conversion works in both directions

On 2026-09-27, a connected `FP32 Abs -> Convert -> FP16 Abs` graph isolated
the conversion ABI that the ordinary target validator had previously rejected.
The conversion records kept the unary descriptor order—input at `+0x00` and
output at `+0x28`—but deliberately used unequal element types and byte spans:

| Field | Input | Output |
| --- | ---: | ---: |
| element type | FP32 (`1`) | FP16 (`2`) |
| element count | 32 | 32 |
| dense bit stride | 32 | 16 |
| byte span | 128 | 64 |

Both descriptors were static dense CMX tensors in the same observed address
space with disjoint spans. A separately compiled kernel read the first record
as `float`, wrote the second as `__fp16`, and computed
`(__fp16)(input * 0.5f + 1.25f)`. Replacing conversion range 4 changed all
32 graph outputs to that formula exactly for both a linear vector and an
irregular FP32 vector. Replacing only the otherwise matching range 5 left all
32 outputs at the original conversion result.

**First direction confirmed:** the tested NPU3720 ACT entry convention can
carry a static dense FP32 input and static dense FP16 output with equal element
counts but different byte spans. Custom code can perform the conversion inside
a connected graph; the earlier rejection was a limitation of `npunlock`'s
single-precision target model, not evidence that the hardware ABI required
equal spans.

The range-5 control establishes only that this range was not observable at the
chosen graph output under this schedule. It does not prove that the invocation
was skipped. More importantly, two consecutive invocations with the same
current group-identity bytes did not behave as two output partitions. Their
element counts therefore must not be blindly summed or treated as an exact
cover.

The carrier blob SHA-256 was
`5bc0ab6845b081782c35b0e356aac75c5d674efbe77621a10c097cd2503830d3`.
The linked kernel ELF SHA-256 was
`31d6b47d29e12569245d1220836bfdeb6b6d8d223d29a782c94f1b2dd12726e2`;
its 208-byte executable image SHA-256 was
`f2f369526d2251a012abd1f797ea93b11b4259b8d40bcb4176daf4169c154f40`.

A reverse experiment used the connected graph
`FP16 Abs -> Convert -> FP32 Abs`. Its conversion records mirrored the first
case: 32 FP16 input elements occupied 64 bytes and 32 FP32 output elements
occupied 128 bytes. A custom range read `__fp16`, wrote `float`, and computed
`(float)input * 1.75f - 0.375f`. After the surrounding `Abs` stages, all 32
outputs matched an independent host calculation for both linear and irregular
FP16 vectors.

The reverse controls repeated the same range behavior: replacing range 4
changed all outputs to the custom formula, while replacing only matching range
5 left every output at the original conversion result. This repetition across
opposite conversion directions strengthens the conclusion that the current
invocation-identity bytes are insufficient to identify output partitions.

**Confirmed:** for these two static unary carriers, the observed ACT descriptor
sequence supports unequal input/output types and spans in both FP32-to-FP16 and
FP16-to-FP32 directions. This conversion result alone did not establish
arbitrary mixed types, mixed-dtype binary kernels, or a public patch-selection
contract; the later two-input result is documented in section 13.

The reverse carrier blob SHA-256 was
`ec78179dab71abe81bcf2328d11b0ce6e77e8e2691d3685bba0152dfa299711e`.
The reverse linked kernel ELF SHA-256 was
`cfdda003858ea5336103808db0a60b0771f4642accf7d003236803d85043e04e`;
its 208-byte executable image SHA-256 was
`3d9c6375e491c1c764902004b26045ac27e82f207a225bf46b0a07f051b50e9f`.

At the time of this experiment, `patchblob_target` carried one precision flag
and one expected span, so it could not express separate input and output
contracts. Section 15 records the later versioned per-tensor implementation
and connected public-API proof; the original target remains unchanged.

## 12. Matching ranges can be tile-local replicas

On 2026-10-01, the previously invisible range 5 from both mixed-conversion
carriers was traced through the scheduling records. The 64-byte invocation
layout identifies range 4 as tile 0 and range 5 as tile 1. Both invocations are
included in the mapped-inference task list, wait on barrier 1, and post barrier
2. Barrier 2 declares exactly two producers, consistent with both conversion
invocations participating in the schedule.

Their tensor descriptors reveal two full-tensor tile-local data paths rather
than two output partitions. In the FP16-to-FP32 carrier, for example, range 4
uses CMX addends `0x40 -> 0x80`, while range 5 uses
`0x200040 -> 0x200080`. The downstream ACT operations preserve the same tile
separation. The network-output DMA source points at the final tile-0 span,
which explains why replacing range 5 was not visible in the original graph
output.

A controlled positive experiment changed only the output DMA source relocation
to the corresponding tile-1 span. Three blobs were compared in each precision
direction:

| Output routed from tile 1 | Original carrier | Range 4 replaced | Range 5 replaced |
| --- | ---: | ---: | ---: |
| FP32-to-FP16, linear input | 32/32 baseline | 32/32 baseline | 32/32 custom |
| FP32-to-FP16, irregular input | 32/32 baseline | 32/32 baseline | 32/32 custom |
| FP16-to-FP32, linear input | 32/32 baseline | 32/32 baseline | 32/32 custom |
| FP16-to-FP32, irregular input | 32/32 baseline | 32/32 baseline | 32/32 custom |

Every graph load and execution completed without an NPU-side error. The
unpatched tile-1 output matching the ordinary result verifies that the routing
mutation selected a coherent duplicate path. The range-5 result then proves
that the second conversion invocation executes and consumes its tile-local
descriptors; it was hidden by output routing, not skipped.

**Confirmed for these two carriers:** equal-contract ACT invocations can be
executable tile-local replicas of the same logical tensor. The invocation tile
field and the network-output DMA source together explain which replica is
host-visible.

**Consequence:** replicas must not be treated as output partitions whose
element counts are summed. The later version 2 target model describes
per-record types and spans and replaces the complete explicitly selected
group rather than only whichever tile currently feeds the network output.

**Still open:** whether the same tile relationship and `0x200000` CMX delta
hold across other shapes, operators, compiler versions, or NPU generations;
and how to represent replica sets in the stable public selector. Section 13
tests one mixed-dtype multi-input contract.

## 13. Mixed-dtype two-input ACT kernels work

On 2026-10-01, a static `[1,32]` `GatherElements` graph exposed a genuine
mixed-type two-input ACT invocation. The operator had to be serialized with
its correct opset version; the same graph labeled as opset1 was rejected by
the driver compiler. The accepted carrier used FP32 data and constant I32
indices and produced an FP32 output.

The invocation kept the established consecutive 40-byte descriptor layout:

| Offset | Role | Raw type | Interpreted type | Count and span |
| ---: | --- | ---: | --- | --- |
| `+0x00` | data input | `1` | FP32 | 32 elements, 128 bytes |
| `+0x28` | indices input | `9` | I32 | 32 elements, 128 bytes |
| `+0x50` | output | `1` | FP32 | 32 elements, 128 bytes |

The compatible software-kernel runtime enum identifies raw type 9 as
`NN_I32`. All three descriptors were static, dense, and CMX-resident. As in
the conversion probes, the compiler emitted two full-tensor replicas: range 0
on tile 0 and range 1 on tile 1, separated by a `0x200000` CMX addend.

The replacement kernel used every index to make two non-sequential reads:

```text
j = indices[i]
k = (3 * j + 1) & 31
output[i] = 1.375 * data[j] + 0.125 * data[k]
```

This formula makes an accidental pass from ignoring either input extremely
unlikely. With the ordinary tile-0 output route, replacing range 0 matched all
32 host-oracle values exactly for both a linear and an irregular FP32 input.
Replacing both tile replicas produced the same result. After changing only the
network-output DMA source to tile 1, replacing only range 1 also matched all
32 values exactly for both inputs. Every execution completed without an
NPU-side error.

An independent `Select` probe provided supporting evidence for heterogeneous
records: its FP32 form exposed an I8 condition followed by two FP32 inputs and
one FP32 output, and a custom kernel using all three inputs also matched two
32-element oracles exactly.

**Confirmed for these carriers:** an NPU3720 ACT invocation can present
consecutive inputs with different element types to custom code. The descriptor
order remains inputs followed by output, and ordinary C pointer types can read
the FP32, I32, and I8 records tested here.

**Important limit:** the unmodified `GatherElements` carrier did not match the
operator's host semantics in this configuration. It served only as a
compiler-generated invocation and scheduling envelope; correctness was
established for the replacement kernel against its own independent oracle.
This result does not claim general `GatherElements` support.

The carrier blob SHA-256 was
`f488baa56ced8f7c85104b9e4d1f895792085e3ac318f4e88b7f51add58ed79b`.
The linked kernel ELF SHA-256 was
`fb8381c28a91bbc9e70f1d6459cb5220a4798495f453e79d4c1a92aa5e78d0f5`;
its 256-byte executable image SHA-256 was
`cdb3cff513f856fd2908fafa8d0fb84b2d39240db629d2037cabd4fb779fb01c`.

**Still open:** integer-valued host graph inputs, other integer widths and
signedness, negative or out-of-range indexing, additional layouts, and a
public target model capable of describing each record independently.
`patchblob` continues to reject this carrier because its supported target
contract is intentionally limited to uniform FP16 or FP32 records. Section 14
tests one unequal-count scalar-broadcast contract.

## 14. Scalar broadcast inputs remain scalar

On 2026-10-02, static `[1,32]` `Maximum` graphs were compiled with a second,
host-provided `[1,1]` input and NumPy-style broadcasting. FP32 and FP16
carriers both exposed four ACT invocations. Each invocation described eight
data elements, one scalar element, and eight output elements:

| Offset | Role | Count | FP32 span | FP16 span |
| ---: | --- | ---: | ---: | ---: |
| `+0x00` | data input | 8 | 32 bytes | 16 bytes |
| `+0x28` | scalar input | 1 | 4 bytes | 2 bytes |
| `+0x50` | output | 8 | 32 bytes | 16 bytes |

The compiler therefore preserved the broadcast operand as a genuine scalar
descriptor rather than expanding it into a full local tensor. The two tile-0
invocations shared one scalar CMX address, and the two tile-1 invocations
shared the corresponding address at the observed `0x200000` tile delta.

The replacement kernel used both inputs for every output:

```text
output[i] = 1.375 * data[i] - 0.625 * scalar[0]
```

After all four ranges selected the replacement image, FP32 and FP16 execution
each matched all 32 host-oracle values exactly for a linear and an irregular
input. Incremental replacements also mapped the range schedule:

| Replaced ranges | Output positions matching the custom oracle |
| --- | --- |
| 0 | 0–7 |
| 0, 1 | 0–7 and 16–23 |
| 0, 1, 2 | 0–23 |
| 0, 1, 2, 3 | 0–31 |

Invocation records identify ranges 0 and 2 as tile 0, and ranges 1 and 3 as
tile 1. The output slices show that these four invocations are partitions,
interleaved by tile in range order, rather than full-tensor replicas.

The unmodified compiler-generated `Maximum` implementation produced
invalid-looking values in this configuration. As with the earlier
`GatherElements` carrier, the graph is used only as an invocation and
scheduling envelope; correctness is established against the replacement
kernel's independent oracle.

Evidence hashes:

```text
FP32 carrier blob  08f2e5055e7d7fa6735188625d527d7c27af06613ba5790136afe538168e48dd
FP32 kernel ELF    24f134a5219e32765910808c5607d251d16adb2c47a0e23dc178f7983f70411e
FP32 patched blob  d72413dcc6f51a3708265c573fc8006a6b40b9cfe27a9e7d4180cb92fe8c693b
FP16 carrier blob  1c1bd31fb53d9338feca495739015d09bf82aae37486b36c2dcaed4d67ed90df
FP16 kernel ELF    2eaa6dd0c31ce3583aa90b8a55ebfaeaec139efade844fd7d9a8349e5e26d2b0
FP16 patched blob  b3b2e4097d7408e8451780e5840e1c38ca05f49ed16f3d79c6d538f6ab8b2a50
```

**Confirmed for these two carriers:** unequal descriptor counts are valid,
and custom code can reuse a one-element input across every element of an
invocation-local output chunk.

`patchblob` now represents this exact layout with the
`PATCHBLOB_CONTRACT_INPUT_1_SCALAR` bit. The ordinary element count and span
continue to describe input 0 and the output; validation requires input 1 to be
one same-precision element. The stable structure did not need to change, and
older uniform contracts remain unchanged.

**Still open:** non-scalar broadcasting, different input positions, axes and
ranks, other shapes and operators, other compiler versions, and a general
per-record type/count/span target model. This narrowly implemented contract
does not enable general broadcasting support.

## 15. Per-tensor contracts enable connected conversion kernels

On 2026-10-04, `patchblob_target_v2` made every validated invocation tensor an
explicit public record: input/output role, index, FP16/FP32 precision, element
count, byte span, and observed static/dense/CMX flags. The original target ABI
was retained unchanged and continues to reject unequal precision.

The first implementation deliberately accepts mixed precision only for unary
FP32-to-FP16 and FP16-to-FP32 records with equal element counts. Discovery
returns both compiler-generated tile replicas in one positional group. Python
exposes their separate input/output dtypes and requires an explicit
prepared-graph binding, avoiding both guessed source mapping and erroneous
partition summation.

A connected `[1,16]` graph exercised both directions:

```text
FP32 input
  -> custom FP32-to-FP16 conversion
  -> ordinary FP16 Abs
  -> custom FP16-to-FP32 conversion
  -> FP32 output
```

The first kernel computed `(__fp16)(x * 0.5f + 1.0f)`. The second computed
`(float)x * 2.0f - 0.25f`. Both targets in each conversion group selected the
same replacement image. The graph executed without an NPU-side error and all
16 final FP32 values matched a NumPy oracle exactly.

Evidence from the successful run:

```text
device                 0x7d1d
driver                 0x000f57e4
graph compiler         8.3
carrier blob SHA-256   6f07ea9ce3eca512c62087356e5a55ad73eb7ba29d05a5a75a720bdca89f6316
FP32->FP16 ELF SHA-256 a149409eb5efa60c1ca42bf7cf51ec21b1098679acb18f4439aa671eae85fce0
FP16->FP32 ELF SHA-256 2cecb079381eca8d91f086a549547c0638ee874157727cf1a2dded352cc67836
patched blob SHA-256   6b7a313be113eb801a8435b9977504fa7a4de7145418a4188880e8e4a49b6b75
```

The runnable proof is
[`example_conversion_kernels.py`](../examples/example_conversion_kernels.py).
This result supports only the stated unary conversion contract. It does not
establish arbitrary mixed-dtype arity, integer records, element-count changes,
broadcasting, or automatic conversion placement.

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
