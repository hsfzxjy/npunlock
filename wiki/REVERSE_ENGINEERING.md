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

## 16. Connected mixed-precision custom and DPU composition

On 2026-10-07, one `[1,32]` graph composed three custom kernels and ordinary
DPU work in a single path:

```text
FP32 host input
  -> custom FP32-to-FP16 conversion
  -> custom two-input FP16 blend, with a second host input
  -> ordinary FP16 MatMul
  -> custom FP16-to-FP32 conversion
  -> FP32 host output
```

Graph compiler 8.3 emitted exactly three positional ACT groups, each with two
invocations. The conversion groups retained version 2 per-tensor contracts.
The middle group had equal FP16 counts and spans for both inputs and its output,
so the frontend could losslessly express it through the established version 1
two-input contract. The native v1 validator still checked the graph records
before patching. Nonempty `.text.DPUInvariants` (`0x260` bytes) and
`.text.DPUVariants` (`0x80` bytes) remained in the carrier for the matrix work.

One prepared carrier and its reviewed symbolic bindings were exported and
loaded again without graph compilation. Three kernels were patched
sequentially through v2, v1, then v2. The original and restored plans each ran
two distinct input sets, and the exported final program ran a third after
reload. All five executions matched an independently rounded NumPy oracle
exactly.

Evidence from the retained run:

```text
device                 0x7d1d
driver                 0x000f57e4
graph compiler         8.3
carrier blob SHA-256   0c491b1afa4b16bd307cb9065b1687779c590ef457cab74ba8469e6986e16de3
FP32->FP16 ELF SHA-256 a149409eb5efa60c1ca42bf7cf51ec21b1098679acb18f4439aa671eae85fce0
FP16 binary ELF SHA-256 e53c09c5938ac22a5bc24ef6f597281b75abf25234d20d09caa973109adc71da
FP16->FP32 ELF SHA-256 2cecb079381eca8d91f086a549547c0638ee874157727cf1a2dded352cc67836
patched blob SHA-256   c45a5d8f095fa57ef99886d8e8bd529b96fa4e6d80263e8cfd5dd44c5d4b47b9
```

The runnable proof is
[`example_connected_mixed_precision.py`](../examples/example_connected_mixed_precision.py).
This establishes composition only for these already validated contracts on the
tested NPU3720/compiler pair. It does not establish arbitrary mixed-precision
graphs, mixed-dtype binary kernels, automatic placement, or general matrix
support.

## 17. I32 graph I/O and structural graphs

On 2026-10-12, graph compiler 8.3 accepted static I32 graph arguments on the
tested NPU3720. Level Zero reported device precision `0x05` (`INT32`), and the
existing bounded allocation path was extended to size those arguments as
four-byte elements. This extends ordinary graph execution, not the custom ACT
descriptor contract.

Three independent device checks passed:

- a `[1,32]` I32 Abs graph matched 32 copied and 32 explicitly shared results;
- a two-input I32 `Concat -> Transpose -> Abs` graph matched all nine output
  elements; and
- an FP32 `Convert(destination_type="i32") -> Abs` graph matched all 16 I32
  outputs for exactly representable integer-valued inputs.

A further public example returned two outputs from one graph: the I32 result
of `Concat -> Transpose` and an FP32 conversion of that result. Both nine-
element outputs matched NumPy exactly. The graph frontend now infers static
contracts for `Concat`, constant-order `Transpose`, and `Convert`; that
inference is only a convenience; the Intel compiler remains the final operator
authority.

The runnable proof is
[`example_i32_structural.py`](../examples/example_i32_structural.py). These
results establish static dense I32 host graph arguments for the tested
compiler/device and operations. They do not establish I32 ACT descriptors,
integer custom kernels, dynamic shapes, every integer operation, or the other
integer widths listed by the Level Zero extension.

## 18. U8 graph I/O and static Reshape

On 2026-10-12, graph compiler 8.3 accepted a static U8 input and output on the
tested NPU3720. Level Zero reported device precision `0x04` (`UINT8`), and the
bounded allocation path sized those arguments as one-byte elements. As with
I32, this is ordinary graph execution and does not extend the custom ACT
descriptor contract.

Two device checks used the boundary values `0`, `1`, `127`, `128`, `254`, and
`255`:

- a `[2,3] -> [3,2]` Reshape matched NumPy exactly through both copied and
  explicitly shared input/output buffers; and
- a two-output graph returned the reshaped U8 tensor and its FP32 conversion,
  with every element matching the corresponding NumPy oracle.

The first carrier was 23,416 bytes with SHA-256
`6d2bb701ff5d7c98209b35223ebcbbfffab90872e24c1c67f8e62143b41d9a4f`.
The frontend now infers a static Reshape contract only from a one-dimensional
integer constant pattern. It validates total elements, one optional `-1`, and
explicit `special_zero` semantics. Dynamic patterns still require unsupported
dynamic graph contracts.

The runnable proof is
[`example_u8_reshape.py`](../examples/example_u8_reshape.py). These results do
not establish U8 ACT descriptors, U8 custom kernels, arbitrary quantized
networks, every U8 operation, or dynamic shapes.

## 19. I8 graph I/O and static axis transforms

On 2026-10-12, graph compiler 8.3 accepted static signed-I8 graph arguments on
the tested NPU3720. Level Zero reported precision `0x07` (`INT8`), and the
bounded execution path sized it as one byte per element. This is ordinary graph
I/O only; the custom ACT contract remains FP16/FP32.

One graph took a `[1,2,1,3]` I8 input, applied positive-axis `Squeeze`, restored
the original shape with `Unsqueeze`, and returned a parallel FP32 conversion of
the squeezed result. Copied and explicitly shared executions both matched
NumPy exactly for `-128`, `-127`, `-1`, `0`, `1`, and `127`. The initial
compiler probe produced a 79,000-byte carrier with SHA-256
`6a95a4e47964b2f4c765f917559f3fdc0ed1b850dec36e0335f3e59ab4c8d71c`.

The frontend now infers `Squeeze` and `Unsqueeze` output shapes from a
one-dimensional integer constant, validates uniqueness and bounds, and
normalizes negative axes. That frontend calculation does not prove compiler
acceptance: graph compiler 8.3 rejected the otherwise equivalent `[0,-2]`
Squeeze axes as a duplicate axis while accepting `[0,2]`. The verified device
path therefore uses positive axes.

The runnable proof is
[`example_i8_axes.py`](../examples/example_i8_axes.py). These results do not
establish I8 ACT descriptors, I8 custom kernels, quantization semantics,
arbitrary signed-byte operators, or negative-axis compiler support.

## 20. I16 graph I/O and multi-output Split

On 2026-10-12, graph compiler 8.3 accepted a static signed-I16 input and three
signed-I16 outputs for an equal `Split` on the tested NPU3720. Level Zero
reported precision `0x06` (`INT16`), and the bounded graphinfer allocation path
sizes it as two bytes per element. This remains ordinary graph execution, not a
custom ACT-kernel contract.

The tested graph split a `[2,6]` input into three `[2,2]` outputs along a scalar
constant axis. Copied and explicitly shared execution matched three NumPy
slices exactly for data including `-32768` and `32767`. The carrier was 43,168
bytes with SHA-256
`5abd0045fb857bd93d5a0df97765762b205d3a3e9501a965e31f26a6e389f99b`.

This experiment also exposed and fixed a general frontend issue. The compiler
names multi-output ports from their source, for example `split.0`, while the
frontend previously invented `Result_0`. Copied execution happened to pair
outputs by order, but shared execution correctly rejected the non-matching
exact-name selectors. Program contracts now derive the same source-and-port
names as the serialized graph, allowing every shared output to bind exactly.

An adjacent I16-to-FP32 Convert probe was rejected by graph compiler 8.3 as an
unsupported `si16 -> f32` configuration. No such conversion path is claimed.
The runnable positive proof is
[`example_i16_split.py`](../examples/example_i16_split.py). It does not
establish I16 ACT descriptors, arbitrary I16 operators, uneven splits, dynamic
axes, or general multi-output operator support.

## 21. U16 graph I/O and mixed-input Gather

On 2026-10-12, graph compiler 8.3 accepted an equal two-way `Split` with one
static U16 input and two U16 outputs on the tested NPU3720. Level Zero reported
precision `0x03` (`UINT16`), which extends the ordinary graphinfer allocation
path with checked two-byte element sizing. Copied and explicitly shared
execution matched NumPy exactly for values from `0` through `65535`. The
33,296-byte carrier had SHA-256
`8ad4bd8254a3079591c8cbd8ce181bb5df7a2b059f733c4e8f37cd8dac684a5c`.

A separate `Gather` graph used a U8 `[3,4]` data input, caller-provided I32
`[2]` indices, and a scalar constant axis. Both copied and shared execution
matched `numpy.take` exactly, including U8 boundary values and reordered rows.
Its 58,408-byte carrier had SHA-256
`0aa11f12cfdead73d07b7f6c08fe33f09ceb861f33abfe5ee3b124027d8bb70d`.
The frontend therefore infers static `Gather` output contracts only for a
constant scalar axis and `batch_dims=0`; compiler acceptance remains the final
authority for each data dtype and shape.

The failed adjacent probe is equally important: the same Gather shape with
U16 data compiled and submitted successfully, but every output value was
incorrect. A successful fence is not a semantic oracle, so U16 Gather is not
claimed. The positive runnable proofs are
[`example_u16_split.py`](../examples/example_u16_split.py) and
[`example_u8_gather.py`](../examples/example_u8_gather.py). They do not
establish U16 ACT descriptors, arbitrary U16 operators, batched Gather,
dynamic axes, out-of-range-index behavior, or general mixed-dtype graphs.

## 22. U32 graph I/O

On 2026-10-12, graph compiler 8.3 accepted a static U32 `[2,6]` input and three
U32 `[2,2]` outputs for an equal Split on the tested NPU3720. Level Zero
reported precision `0x0a` (`UINT32`), and graphinfer now uses checked four-byte
element sizing for that value.

Copied and explicitly shared execution both matched three NumPy slices exactly
for values including `0`, `2**31 - 1`, `2**31`, `2**32 - 2`, and `2**32 - 1`.
The 43,168-byte carrier had SHA-256
`5d8b11f799a89d470207e3969cbf4ffed412665813da8705aa1f8de55c80997f`.
The runnable proof is
[`example_u32_split.py`](../examples/example_u32_split.py).

This establishes ordinary U32 graph transport for the tested structural graph.
It does not establish U32 arithmetic operators, conversion paths, custom ACT
descriptors, uneven splits, dynamic axes, or support outside NPU3720.

## 23. I64 Gather indices and the I64 data boundary

On 2026-10-12, a U8 Gather graph with a caller-provided I64 `[2]` indices input
executed correctly on the tested NPU3720. Copied and explicitly shared paths
both matched `numpy.take` exactly for reordered rows and U8 boundary values.
The 85,064-byte carrier had SHA-256
`bc906d4073af22d1ccc0e8f8e5a78c8ac65892c7f3c0397866fca93e61de29c5`.
Level Zero reported precision `0x11` (`INT64`) for the index argument, so
graphinfer uses checked eight-byte element sizing for that restricted role.

An adjacent I64-data Split established a sharp negative boundary. Small signed
values happened to round-trip, but values outside the signed 32-bit range were
silently narrowed or truncated. For example, `2**31` returned as
`-2147483648`, and `2**63 - 1` returned as `-1`. A shared run containing the
extreme values also once returned a device error at fence synchronization.
Compiler acceptance and successful submission therefore do not establish
full-width I64 data support.

The runnable positive proof is
[`example_u8_gather_i64.py`](../examples/example_u8_gather_i64.py). The public
contract accepts I64 only as an index input for the validated Gather shape; it
does not claim I64 graph outputs, arithmetic, conversion, or custom ACT
descriptors.

## 24. Unequal multi-output VariadicSplit

On 2026-10-12, graph compiler 8.3 accepted a U8 `[2,7]` input split into
unequal `[2,2]`, `[2,1]`, and `[2,4]` outputs by a constant-axis
`VariadicSplit`. Copied and explicitly shared execution both matched NumPy
slices exactly, including `0` and `255`. Exact compiler-compatible output names
(`parts.0`, `parts.1`, and `parts.2`) bound each shared allocation without
falling back to positional guessing.

The carrier was 43,168 bytes with SHA-256
`a1fa976e110311baa98c325bfdfa3c94178901e4198581c4fad4e6b8c6c22e9a`.
The frontend now derives static output contracts from a scalar constant axis
and one-dimensional constant lengths. It accepts positive lengths and at most
one `-1` remainder only when the resolved lengths exactly cover the selected
input dimension.

The runnable proof is
[`example_u8_variadic_split.py`](../examples/example_u8_variadic_split.py).
This does not establish dynamic split lengths, zero-length outputs, multiple
inferred lengths, arbitrary output counts, or non-U8 compiler support.

## 25. Static positive-step Slice

On 2026-10-14, graph compiler 8.3 accepted a U8 `[3,8]` input sliced over two
axes with constant starts `[1,1]`, stops `[3,8]`, steps `[1,2]`, and axes
`[0,1]`. The resulting `[2,4]` output matched NumPy exactly through both copied
and explicitly shared execution.

The first probe was rejected because the frontend serialized every ordinary
operation as `opset1`. `Slice` requires its later IR declaration; emitting it
as `opset8` produced a 23,416-byte carrier with SHA-256
`ac31959684192f3d5c07b923f2b4b615cc467a3f3662d3e03c1808ff686f00c4`.
This is a frontend serialization rule, not a native graph ABI change.

The frontend now infers static Slice contracts only from one-dimensional
integer constants. It accepts unique in-range axes and positive steps, and
normalizes negative axes and Python-style negative bounds. The runnable proof
is [`example_u8_slice.py`](../examples/example_u8_slice.py). This result does
not establish dynamic bounds, zero or negative steps, empty outputs, arbitrary
strides/layouts, non-U8 execution, or a general mapping from operator names to
IR opset versions.

## 26. Static constant-mode Pad

On 2026-10-14, graph compiler 8.3 accepted a U8 `[2,3]` input padded to
`[4,7]` by begin pads `[1,2]`, end pads `[1,2]`, and the scalar U8 value `255`.
Copied and explicitly shared execution both matched `numpy.pad` exactly,
including every filled border element.

The carrier was 25,224 bytes with SHA-256
`f7ca31d4663395dc6a69178ceff0272abb58387464582fc1f206f1d0bff0c368`.
The frontend now infers this contract only when both pad vectors are constant,
nonnegative, and contain one value per input axis; the fill must be a constant
scalar with the same dtype as the data. The runnable proof is
[`example_u8_pad.py`](../examples/example_u8_pad.py).

This result does not establish dynamic padding, negative cropping, reflect or
edge modes, non-scalar fill values, non-U8 execution, or custom ACT-kernel
padding.

## 27. Row broadcast at the ACT descriptor boundary

On 2026-10-14, graph compiler 8.3 accepted independently bound FP16 inputs
shaped `[4,8]` and `[1,8]` for a custom binary `Maximum` carrier. Discovery
reported one group containing four invocations. Every invocation exposed two
dense eight-element inputs and one dense eight-element output: `(8,8) -> 8`.
The compiler therefore presented the host row broadcast as a complete dense
operand to each custom entry rather than exposing a scalar contract or a
separate indexing rule. This establishes descriptor-level availability, not
whether the backing bias storage is physically copied or reused.

A custom entry computing `half(data[i] + bias[i])` matched NumPy's row
broadcast exactly through copied and explicitly shared inputs and outputs. The
64,808-byte unpatched carrier had SHA-256
`87fd03faec4e4ee1bf97bd2e7c77b44286082d23c7c81d98c24b47ef1a1e5ebc`.
With the current overflow-checked target header, the linked 924-byte SHAVE ELF
had SHA-256
`d78d5a472a887579ad868ef3e72b061996d74e6a269c451fcfe0aa433530e262`,
and the 65,400-byte patched graph had SHA-256
`78f788d42e8c748b55231e28631ab76c235db4430e6f0697907965fa36db069f`.

`ActGroup` and `PreparedGraph.explain()` now expose stable per-invocation input
and output element counts so kernel authors can distinguish a complete dense
operand descriptor from the existing scalar-second-input contract before
patching.
The runnable proof is
[`example_broadcast_custom.py`](../examples/example_broadcast_custom.py).

This establishes only this FP16 carrier, shapes, compiler version, and dense
descriptor presentation. It does not establish arbitrary broadcasting,
per-channel layouts, other ranks/dtypes, or permission to read beyond any
invocation's advertised descriptor span.

## 28. Five-input custom kernel

On 2026-10-14, graph compiler 8.3 produced a patchable FP16 `FakeQuantize`
carrier from five independently bound `[1,16]` graph inputs. Discovery found
one group with four invocations. Each invocation contained five consecutive
dense eight-element input descriptors followed by one disjoint eight-element
output descriptor: `(8,8,8,8,8) -> 8`.

A custom entry read records 0 through 4 and wrote record 5. It weighted the
five operands by distinct powers of two so every input affected independently
distinguishable output bits. Copied and explicitly shared execution both
matched the exact NumPy oracle for all 16 graph outputs.

The unpatched carrier was 95,080 bytes with SHA-256
`abf9e903b79ca5fb3e4cc3f692b16d8d1ef0150c6fd4c314183b2fe06b15fc70`.
The linked 1,020-byte SHAVE ELF had SHA-256
`eca5e4be08aad433fac48ba3ec23966ad3aa8b898b3bd7bcff23bfa527934c25`,
and the 96,104-byte patched graph had SHA-256
`1c767cdc3e5dff8360e8c6a373662627e31bd38fb5fb9508b93187d8e508e016`.
The runnable proof is
[`example_five_input_custom.py`](../examples/example_five_input_custom.py).

This establishes one static dense FP16 five-input layout. It does not establish
arbitrary arity, unequal five-input spans, other dtypes, `FakeQuantize`
semantics after patching, or a general rule that every high-arity graph
operation becomes an ACT carrier.

## 29. Structured kernels and invocation-local reductions

On 2026-10-14, progressively more structured FP16 entries established that a
custom kernel need not remain one flat elementwise loop. The first retained a
distinct same-file helper, branched on the local index, read a clamped neighbor,
and used a source-level 128-byte scratch array. The second executed a nested
three-helper call chain, passed a local context structure, used a source-level
256-byte scratch array, and read a retained 20-byte coefficient object while
applying a five-tap invocation-local window. Both copied and shared execution
matched exact chunk-aware NumPy oracles.

The third entry performed a maximum reduction over each invocation, evaluated
`expf(input[i] - maximum)` into scratch, accumulated a sum reduction, and
normalized the output in a final pass. A logical 64-element tensor was exposed
as 32-element chunks, so the result is two independent local normalizations,
not graph-global softmax. Copied and shared FP16 results matched NumPy within
`rtol=5e-3`, `atol=5e-4`; the deterministic proof input had zero observed FP16
error.

The softmax ELF was 53,796 bytes with SHA-256
`6bda0666bd958565a85ed8e39cb6579a3229fc28da4e66f7855bead838f18e07`,
but most bytes were non-loadable debug information. Its executable `.text`
image was 976 bytes, `.arg.data` was empty, and no relocations or undefined
symbols remained. This confirms that `patchblob` must install the validated
load image rather than treating the complete ELF file size as kernel code.

The runnable proofs are
[`example_complex_kernel.py`](../examples/example_complex_kernel.py),
[`example_window_kernel.py`](../examples/example_window_kernel.py), and
[`example_invocation_softmax.py`](../examples/example_invocation_softmax.py).
These results do not establish recursion, arbitrary call graphs or stack
sizes, cross-invocation neighborhoods, or graph-global reductions.

## 30. Data-dependent loops and local permutation

On 2026-10-15, a custom FP16 entry loaded each ACT invocation into a bounded
FP32 scratch array and ran insertion sort. Its inner `while` loop had a
data-dependent trip count, shifted values through indirectly indexed scratch
positions, and emitted a permutation rather than an elementwise mapping.

The logical 64-element tensor was exposed as 32-element invocation chunks.
Copied and explicitly shared execution both exactly matched two independent
NumPy sorts over those boundaries. The linked 1,244-byte ELF had SHA-256
`811bce380fb541159cd6fe090085e48c55b1436dfe1ee980ae7f699503601a42`;
its executable `.text` image was 608 bytes, `.arg.data` was empty, and no
relocations or undefined symbols remained. The runnable proof is
[`example_invocation_sort.py`](../examples/example_invocation_sort.py).

This establishes bounded data-dependent local loops, indirect scratch access,
and local value permutation. It does not establish graph-global sorting,
cross-invocation movement, index outputs, dynamic allocation, or safe
unbounded loop runtimes.

## 31. Fused five-input local RMS normalization

On 2026-10-15, the validated five-input FP16 `FakeQuantize` carrier ran a
single composed entry using every descriptor. The kernel added data and
residual inputs into FP32 scratch, accumulated an invocation-local sum of
squares, used `sqrtf` to compute an inverse RMS, then applied independent
per-element scale, bias, and gate inputs.

The carrier retained four `(8,8,8,8,8) -> 8` invocations. Copied and explicitly
shared execution matched a chunk-aware NumPy oracle within `rtol=5e-3` and
`atol=5e-4`; the deterministic proof input had zero observed FP16 error. The
linked ELF was 53,640 bytes with SHA-256
`d9474290a6f8f4553e8f13bcf262fda40160e2fd7974f1e63e921bb2bf31f9a7`.
Its executable `.text` image was 864 bytes, `.arg.data` was empty, and no
relocations or undefined symbols remained. The runnable proof is
[`example_fused_rmsnorm.py`](../examples/example_fused_rmsnorm.py).

This confirms that high arity, local reduction, mlibm math, scratch reuse, and
per-element post-processing can coexist in one custom entry. It does not
establish standard graph-global RMSNorm semantics, arbitrary five-input
carriers, unequal spans, or other precisions.

## 32. Graph shape and invocation shape are distinct

On 2026-10-15, a metadata probe compiled from a logical FP16 input shaped
`(4,8)` exposed one unary ACT group with four eight-element invocations. Each
custom entry reported rank 4 and dimensions `(8,1,1,1)`, rather than the
logical rank-2 graph shape. The element count remained the checked product of
those invocation dimensions.

A follow-up entry required exactly that rank/layout, obtained its width through
`act_abi_dimension()`, and reversed the eight local values. Copied and
explicitly shared execution exactly matched reversal of each of the four
host-visible rows. The linked ELF was 1,004 bytes with SHA-256
`a44511b2311bd2d947bf54a4d92c3a7fbffb7c53822ce6cb43296d81bd72a425`;
its executable `.text` image was 368 bytes, `.arg.data` was empty, and no
relocations or undefined symbols remained. The runnable proof is
[`example_dimension_aware.py`](../examples/example_dimension_aware.py).

This establishes that kernels can validate and use the observed invocation
dimensions. It does not establish a general mapping from graph axes to ACT
axes, or the same layout for other shapes, carriers, compiler versions, or NPU
generations.

## 33. Bounded indirect helper dispatch

On 2026-10-15, a custom FP16 entry selected between two same-file non-inlined
transforms from each invocation value. It stored the selected address in a
volatile local function pointer and passed that pointer into a separate
non-inlined dispatcher. Negative and nonnegative inputs exercised both targets
within the same logical tensor.

The linked ELF retained 44- and 48-byte transform symbols plus a 38-byte
dispatcher. It was 1,344 bytes with SHA-256
`960d85077df29fc35338f229d71c4242673c5789f388a7ea4a1b96e9732181db`;
its executable `.text` image was 608 bytes, `.arg.data` was empty, and no
relocations or undefined symbols remained. Copied and explicitly shared
execution exactly matched the chunk-aware NumPy oracle. The runnable proof is
[`example_function_dispatch.py`](../examples/example_function_dispatch.py).

This validates one bounded indirect-call shape with fully linked same-file
targets. It does not establish global function-pointer tables, pointers read
from tensor data, external targets, callbacks, recursion, or arbitrary
indirect call graphs.

## What remains deliberately unresolved

The experiments above reconstruct a useful path, not a complete Intel NPU SDK.
The project does not currently claim support for:

- dynamic shapes, arbitrary strides or layouts, or precisions beyond the
  documented ordinary graph and custom-kernel paths;
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
