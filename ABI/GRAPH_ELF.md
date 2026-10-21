# NPU3720 native graph ELF: observed ABI

Status: experimental. Last updated 2026-10-19.

This document explains the part of an Intel NPU3720 native graph blob that
`npunlock` reads and modifies. It is written for readers who know ordinary ELF
files but may not know Intel NPU terminology.

This is an observed compatibility contract, not an Intel specification. The
document covers only structures that were validated on a Meteor Lake NPU
(device `0x7d1d`) with graph extension 1.18 and compiler 8.3.

The confidence words used below have precise meanings:

- **Confirmed**: byte inspection and controlled NPU execution agree.
- **Inferred**: several observations agree, but the official field definition
  is not available.
- **Unknown**: `npunlock` must not rely on the behavior.

## What a native graph blob contains

The NPU compiler turns a model into much more than kernel machine code. The
native graph also contains tensor placement, DMA work, synchronization,
hardware scheduling, kernel parameters, and relocations between those pieces.

For a custom activation (`ACT`) kernel, the important relationship is:

```text
invocation record
  -> parameter block for this invocation
  -> range record describing which code to run
  -> code bytes inside KernelText
```

One logical graph operation may have several invocation records because the
compiler partitions a tensor into chunks. Several range records may also point
to the same code bytes.

`npunlock` preserves the compiler-generated scheduling and tensor metadata. It
changes only the code selected by validated ACT range records.

## ELF container

**Confirmed:** the native graph is an ELF64, little-endian, section-oriented
container. It resembles a relocatable ELF file, but some identity fields do
not describe a normal host CPU:

| ELF field | Observed value |
| --- | ---: |
| class | 2 (`ELFCLASS64`) |
| data encoding | 1 (little-endian) |
| `e_type` | 1 (`ET_REL` value) |
| `e_machine` | 0 |
| ELF version fields | 0 |
| entry point | 0 |
| program headers | none |
| section-header offset | `0x40` |
| section-header size | 64 bytes |

A parser therefore cannot apply ordinary x86-64 ELF identity checks. It must
validate the observed NPU container fields and then bounds-check every section
and relocation it uses.

`npunlock` requires this exact section-only identity, including System V ABI
value 0, zero ELF version fields, zero flags, a 64-byte ELF header, and zero
program-header offset/count/entry size. A familiar section name is not enough
to make an ELF with a different type, machine, entry point, or header layout a
supported native graph.

The supported section table is also deliberately narrow. Section zero is the
observed null record (all semantic fields zero, alignment 4); section-name
table index 1 names `.strtab`; and file-backed sections must be disjoint. Every
nonzero section alignment must be a power of two, and each file offset must
satisfy it. The required ACT sections retain their observed flags, alignment,
link, info, and entry-size fields—for example, `KernelText` has alignment
1024, while parameters, ranges, and invocations have alignment 64. These checks
prevent familiar names in a malformed section table from reaching the
vendor-record parser.

Common sections include:

```text
.text.KernelText             ACT machine code
.text.KernelData             optional ACT data
.text.KernelParams           per-invocation parameters and tensor descriptors
.text.ActKernelRanges        code selections and extents
.text.ActKernelInvocations   scheduled ACT work
.text.dmaTasks*              data movement
.text.BarrierConfigs         synchronization
.text.DPUInvariants          DPU configuration
.text.DPUVariants            DPU work partitions
.text.MappedInference        top-level execution description
.metadata                    graph metadata
.rlt.*                       NPU relocation tables
.symtab.*                    symbol tables
```

The observed blobs also contain a vendor `compatibility_string` section with
section type `0x8aaaaab2`. Its payload is a NUL-terminated,
semicolon-delimited byte string such as:

```text
compiler=8.3;npu=3720;t=2;elf=1.2.0;mi=7.0.4
```

**Confirmed:** compiler 8.3 emits this section in the retained unary, binary,
same-precision, and conversion carriers, and the compiler and NPU fields agree
with the versions queried from the driver at graph creation. Across 76 local
blob artifacts, `t=1` bounded ACT invocation tile indices to 0, while `t=2`
bounded them to 0 and 1. This correlation supports treating `t` as a tile-count
bound; it does not say whether two invocations are partitions or replicas.

The advertised `elf=1.2.0` and `mi=7.0.4` values match the dedicated note
sections:

| Section | Owner/type | Four 32-bit description words |
| --- | --- | --- |
| `.note.LoaderABIVersion` | `GNU`, type 1 | `0, 1, 2, 0` |
| `.note.MappedInferenceVersion` | `MIV`, type `0xa000` | `0, 7, 0, 4` |

`npunlock` requires these observed compiler, NPU, one-or-two-tile, loader-ELF,
and mapped-inference contracts before interpreting ACT records. Missing,
duplicate, inconsistent, or different fields are rejected rather than treated
as a similar-enough layout. Unknown extra compatibility fields are preserved
without assigning them semantics.

The `.meta.PlatformInfo` section independently identifies the architecture.
For NPU3720 it has vendor section type `0x8aaaaaaf`, size 40, and alignment 8.
Its bytes describe one eight-byte serialized element:

```text
data offset       0x20
next descriptor  0
element count     1
element size      8
architecture      3 (NPU3720)
```

The value 3 is the loader platform enumeration for NPU3720. All 78 current
blob artifacts inspected used this exact descriptor. `npunlock` cross-checks
it against `npu=3720`; it does not interpret other platform records as
compatible.

The exact section set, order, size, and file offsets vary by graph. Absolute
file offsets are not an ABI. `npunlock` follows section references and
relocations instead.

### DMA task records

For the NPU3720/compiler-8.3 contract, each `.text.dmaTasksN` section contains
128-byte task records and `N` identifies the DMA engine. The fields currently
used by the inspector are:

| Record offset | Size | Observed meaning |
| ---: | ---: | --- |
| `0x08` | 8 | configuration; low two bits select the transfer record form |
| `0x10` | 8 | source address, resolved by relocation |
| `0x18` | 8 | destination address, resolved by relocation |
| `0x20` | 4 | transfer length in bytes |
| `0x24` | 4 | low byte plane count; upper 24 bits task identifier |
| `0x28`, `0x2c` | 4 each | signed source and destination plane strides |
| `0x30..0x3f` | 16 | type 0 barrier masks, or type 1 width/stride geometry |
| `0x40`, `0x48` | 8 each | type 1 post and wait barrier masks |
| `0x50`, `0x54` | 4 each | start-after and clean-after scheduling values |

Every accepted task has exactly one relocation for its source field and one
for its destination field. Relocation type 0 (`R_VPU_64`) writes a relocated
64-bit address; relocation type 1 (`R_VPU_64_OR`) preserves the record's raw
address bits while combining them with the relocated base. Relocation symbol
tables distinguish
CMX runtime addresses, graph inputs and outputs, constants, and intermediate
DDR buffers. The inspector reports the unrelocated field value, relocation
type, symbol and addend so callers do not have to infer memory kinds from raw
addresses.

These offsets are confirmed for the stated NPU3720/compiler-8.3 container
contract. They are not claimed for newer NPUs. A chain of compatible DMA
endpoints and barriers is native scheduling evidence, not by itself a semantic
source-graph edge.

The inspector can derive one stricter relationship without assigning source
semantics. An ACT-to-DMA source candidate requires the ACT output and a
contiguous CMX-to-DDR transfer to have the same runtime symbol, exact address
and byte span, plus a direct post/wait barrier intersection. Multiple such DDR
destinations form an exact fan-in only when they are unique, non-overlapping,
have one interval union, and one uniquely scheduled contiguous transfer reads
that complete union into the observed NPU3720 multicast-CMX form. This proves
the bytes and schedule through the DMA chain.

For the narrower one-ACT-invocation-per-tile case, the inspector can continue
this proof to DPU inputs without assuming a fixed tile stride. It subtracts
the tile-0 ACT source address from each tile's corresponding ACT source
address, applies that observed translation to the multicast destination, and
requires one DPU input on the same cluster with the translated address, equal
span, and a direct barrier dependency. Every observed tile must match exactly
once. This proves a native per-tile ACT-to-DMA-to-DPU input relationship; it
does not identify a source operation, establish a general multicast encoding,
or make the group an automatic patch target.

A second derived relationship covers an ACT output that is spilled to DDR and
later reloaded for another ACT group. The spill must be a unique exact
ACT-output-to-DMA source candidate. Each accepted reload is a simple contiguous
DDR-to-CMX task whose source lies inside the same DDR interval and whose
destination exactly matches one later ACT input's runtime symbol, address,
precision, and byte span. Direct barrier dependencies are required across both
the spill-to-reload and reload-to-consumer hops. Reload tasks must be unique,
the reload intervals must form a non-overlapping exact cover of the spill, and
any producer invocation without its own spill must be a unique full-contract
tile replica. This proves a bounded ACT-to-DMA-to-DDR-to-DMA-to-ACT route; it
does not establish source-node identity, a general DDR lifetime, or automatic
patch eligibility.

Observed ACT relocation tables use 24-byte ELF64 `SHT_RELA` entries, flags
`0x40`, alignment 8, and names derived from their target section, such as
`.rlt.text.KernelParams`. Relocation type 4 is `R_VPU_32`: it writes
`S + A` into a four-byte target field. It is used for the ACT code, parameter,
and tensor-address references that `npunlock` follows. This agrees with the
kernel ABI's use of low-32-bit addresses; the 64-bit `r_offset` and `r_addend`
fields in an ELF64 relocation entry do not make the destination field 64 bits.

The supported ACT sections also contain type 3 (`R_VPU_64_LSHIFT`) and type 5
(`R_VPU_32_RTM`) loader relocations in invocation metadata. `npunlock` does not
interpret those fields as custom-kernel pointers, but it validates every entry:
type 3 must name an aligned eight-byte destination, types 4 and 5 must name
aligned four-byte destinations, and the complete write must lie inside the
section selected by `sh_info`. Other relocation types are outside the current
ACT carrier contract.

Two source namespaces occur in the supported compiler-8.3 blobs:

- ordinary section references link to `.symtab.tasks`, whose string table is
  `.symstrtab`; and
- loader-provided addresses use the reserved section index `0xff20`, named
  `VPU_RT_SYMTAB` by the compatible VPU ELF definitions.

The supported ordinary symbols are zero-valued section symbols named
`sym_kernelTextSection`, `sym_kernelDataSection`, or
`sym_kernelParamsSection`, and the name must agree with the referenced
section. Tensor descriptor data pointers use a type-4 write from runtime symbol 0,
`VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR`, plus a CMX offset in the relocation
addend. Runtime symbols 2, 5, and 6 also occur in surrounding invocation
metadata, but `npunlock` does not assign them a custom-kernel contract. Values
outside the observed runtime-symbol range 0 through 7 are rejected.

These names and numbers describe the validated native graph family. They are
not a claim that every VPU ELF producer or newer NPU uses the same namespace.

## ACT invocation records

For the supported compiler family, an `ActKernelInvocation` record is 0x40
bytes.

| Record offset | Observed purpose | Confidence |
| ---: | --- | --- |
| `+0x00` | index of the selected `ActKernelRange` | confirmed |
| `+0x04` | relocated reference to this invocation's `KernelParams` | confirmed |
| `+0x08` | relocated reference to `KernelData` | confirmed; supported custom kernels use no data |
| `+0x10` | barrier wait mask | confirmed against scheduling behavior |
| `+0x18` | barrier post mask | confirmed against scheduling behavior |
| `+0x20` | barrier group and mask bytes | structure-correlated |
| `+0x28` | scheduling `start_after` value | structure-correlated |
| `+0x2c` | scheduling `clean_after` value | structure-correlated |
| `+0x30` | invocation index | confirmed |
| `+0x34` | tile index (`0` or `1` in the tested two-tile graphs) | execution-confirmed |
| `+0x38` | kernel-range index | confirmed |

The number of invocation records is not the number of graph operations. It
depends on shape, layout, compiler partitioning, and tile configuration. For
example, some `[1,16]` FP16 operations have two invocations of eight elements
even when compilation requests one NPU tile.

### Tile-local replicas and output routing

Two tested mixed-precision carriers each contained a pair of full-tensor
conversion invocations rather than two partial-tensor chunks. The records had
the same tensor contract but different tile indices and CMX addends:

```text
range 4  tile 0  base addend +0x000000
range 5  tile 1  base addend +0x200000
```

Both records appeared in the mapped-inference task list, waited on the same
barrier, and posted the same following barrier. That barrier declared two
producers, matching the pair. The final network-output DMA selected the tile-0
CMX result, so replacing range 5 initially produced no host-visible change.

A controlled mutation changed only that DMA source relocation to the
corresponding tile-1 CMX result. The unmodified kernel path still produced the
same baseline output; replacing range 4 remained invisible; and replacing
range 5 then controlled all 32 outputs. This result repeated for FP32-to-FP16
and FP16-to-FP32 conversion with two input sets in each direction.

**Confirmed for these carriers:** matching invocations may be executable
tile-local replicas of the same logical work. An invocation can be active even
when its result is not selected by the graph output. The observed `0x200000`
tile delta and DMA relocation layout are not yet claimed as universal ABI.
Automatic grouping must distinguish replicas from partitions: summing their
element counts would double-count the logical tensor, while patching only the
currently visible replica would depend on compiler routing.

### Repeated DPU input relocations

Some compiler-8.3 graphs contain more than one type-4 CMX relocation for the
same DPU input register. The entries can carry different addends, so the graph
inspector must not infer that the first or last entry is the effective runtime
address. At least one six-invariant compound graph with this structure loaded
and executed successfully, confirming that a repeated target is not by itself
a malformed native graph.

The inspector preserves every observed input relocation and reports a bit for
the repeated register in `input_ambiguous_fields_raw`. Routing proofs ignore
that field entirely; another unique input register in the same invariant may
still participate. Strict comparison includes all retained entries and the
ambiguity mask. This documents uncertainty instead of turning relocation order
into an ABI assumption. Repeated output-register relocations are not yet
accepted.

The same warning also applies to ordinary same-precision groups, not only the
conversion example above. A logical 32-element FP16 unary carrier exposed one
group containing four 16-element invocations. Runtime validation showed that
the group represented two 16-element partitions duplicated across two tiles,
so its aggregate descriptor count was 64. Patching the complete group produced
the correct 32-element result. Neither a sum nor a set of equal invocation
counts can identify logical output coverage without the scheduling and DMA
routing relationship.

```text
invocation 0  tile 0  input +0x000000  output +0x000040
invocation 1  tile 1  input +0x200000  output +0x200040
invocation 2  tile 0  input +0x000020  output +0x000060
invocation 3  tile 1  input +0x200020  output +0x200060
```

## Positional operation groups

The compiler often emits several consecutive invocations for one logical ACT
operation. `patchblob` groups records by comparing the bytes that stayed
constant within an operation across validated compiler outputs. It excludes
known per-invocation fields, range indices, and relocation slots from that
identity comparison.

This distinction matters because two adjacent operations can use identical
machine code while still being separate logical groups. A code address alone
is therefore not a safe operation selector.

Automatic high-level selection is allowed only when all of the following are
true:

1. the computational-node sequence maps uniquely to the positional ACT groups
   in topological order;
2. each ordinary computational node consumes one group;
3. a custom node normally consumes one group with matching input arity and
   tensor precision; and
4. when a large custom node consumes several consecutive groups, every target
   has matching arity, element width and contract flags, invocation indices are
   consecutive, and the summed element count exactly covers its declared
   output tensor.

Graphs containing DPU operations, fused nodes, inserted conversions, or other
ambiguous mappings need an explicit validated target selection. A source node
name is not stored as a reliable range selector in the native blob.

The positional identity slices above are validated only for graph compiler
8.3. The Python frontend refuses discovered positional mapping from another
compiler version. Raw invocation/range targets remain available when the
caller has independently validated a different compiler layout.

## ACT range records

For the supported compiler family, an `ActKernelRange` record is 0x18 bytes.

| Record offset | Observed purpose | Confidence |
| ---: | --- | --- |
| `+0x04` | code virtual address (`0x1d000000`) | confirmed |
| `+0x08` | relocated pointer into `.text.KernelText` | confirmed |
| `+0x0c` | selected code-image size in bytes | execution-tested inference |

The relocation addend at `+0x08` selects a byte offset in `KernelText`. The
extent at `+0x0c` bounds that implementation. More than one range may select
the same code image.

## Kernel parameters and tensor descriptors

Each invocation relocation selects a base inside `.text.KernelParams`. The
supported carriers store tensor descriptors consecutively at that base. Each
descriptor is 0x28 bytes:

| Descriptor offset | Observed meaning |
| ---: | --- |
| `+0x00` | data address; validated kernels use its low 32 bits |
| `+0x08` | raw value 1 for supported static tensors |
| `+0x0c` | rank as `u32` |
| `+0x10` | pointer to `rank` signed 32-bit dimensions |
| `+0x14` | pointer to `rank` signed 64-bit bit-strides |
| `+0x18` | element-type value: 1 for FP32, 2 for FP16, 5 for U32, 6 for U16, 7 for U8, 9 for I32, 10 for I16, and 11 for I8 in observed records |
| `+0x24` | raw value 2 for supported CMX tensors |

The raw element values match the `NN_FP32`, `NN_FP16`, `NN_U32`, `NN_U16`, `NN_U8`,
`NN_I32`, `NN_I16`, and `NN_I8` entries in the compatible software-kernel
runtime type definition. An IR boolean condition was lowered to an I8 descriptor.
`npunlock` still validates the complete combination rather than assuming that
every enum member or mixture is usable.

For a dense tensor, non-singleton dimensions ordered by increasing stride
must start at the element bit width and grow by the previous dimension size.
Singleton strides are ignored because valid compiler outputs do not encode
them consistently. All dimensions must be positive, and every element-count
and byte-span calculation is overflow checked.

The validated descriptor order is:

```text
unary operation:  input at +0x00, output at +0x28
binary operation: input A at +0x00, input B at +0x28, output at +0x50
```

An I8 or U8 span is `element_count`; an FP16, I16, or U16 span is
`element_count * 2`; and an FP32, U32, or I32 span is `element_count * 4`. The output
span must not alias an input span. The original `patchblob_target` ABI requires
all inputs and the output to describe one precision and normally one byte span,
with one explicit scalar-second-input exception. The additive
`patchblob_target_v2` ABI instead records each tensor's role, precision,
element count, span, and observed flags.
Version 2 currently accepts the same established layouts, equal-element unary
FP32-to-FP16, FP16-to-FP32, and U8-to-FP32 conversion, and the observed
same-precision unary I32, I16, and I8 layouts, equal-span two-input integer
layouts, and one observed scalar in either binary input position for
FP16/FP32/U16/U32. The documented unsigned row broadcasts present equal-span
invocation descriptors. This is not a general mixed-dtype, integer, or
broadcasting ABI.

## Validated examples

These examples define the current support boundary; they are not promises
about every graph produced by the compiler.

### Static FP16

Validated unary carriers expose dense invocation-local chunks ranging from 8
to 2048 FP16 elements. Custom kernels successfully read the full advertised
4096-byte input span of a 2048-element chunk. This proves only
invocation-local access. It does not provide a graph-global tensor view or
permission to cross a chunk boundary.

A static `[1,32]` graph with two independent host inputs, separate `Abs`
branches, a binary `Maximum` carrier, and a final `Sqrt` produced four ACT
groups for four computational nodes. The binary group contained four
invocations, each with two eight-element inputs and one output. A custom
weighted-mix implementation matched the host reference exactly.

### Precision-preserved FP32

A plain FP32 `[1,2048]` `Abs` carrier is normally lowered into three ACT
groups: FP32-to-FP16 conversion, FP16 `Abs`, and FP16-to-FP32 conversion.
Replacing only the middle group would lose precision.

Marking the carrier with OpenVINO IR runtime attribute
`DisablePrecisionConversion=dynamic:f16` and compiling with
`EXECUTION_MODE_HINT="ACCURACY"` instead produced one FP32 ACT group. Its four
invocations each described 1024 elements and a 4096-byte input/output span.
A custom FP32 GELU implementation matched a host reference with maximum
absolute error `2.38419e-07`.

This establishes one unary FP32 carrier configuration, not general FP32 graph
support.

### Equal-span binary FP32

With the same precision-conversion suppression and accuracy-mode build flag,
a static two-input FP32 `[1,32]` `Subtract` graph produced one ACT group with
four invocations. Each parameter block used the binary record order input A at
`+0x00`, input B at `+0x28`, and output at `+0x50`. All three records used raw
element type 1, an eight-element count, a 32-byte span, static dense CMX
placement, and a disjoint output.

The unmodified `Subtract`, `Multiply`, and `Add` candidates matched their host
oracles. `Maximum` and `Minimum` exposed the same apparent descriptor contract
but returned corrupt baseline values and were rejected. A substituted
non-commutative `3*a - 2*b + 7` kernel in the retained `Subtract` carrier
matched all 32 FP32 outputs exactly.

This establishes only the tested equal-span `Subtract` carrier on
NPU3720/compiler 8.3. It does not establish other FP32 binary operations,
shapes, unequal spans, or general multi-input FP32 support.

### Same-precision unary I32

A static I32 `[1,32]` `Abs` carrier produced one ACT group containing four
invocations. Each invocation described a dense static CMX input and disjoint
output at the unary `+0x00`/`+0x28` positions. Both records used element-type
value 9, a 32-bit first stride, 16 elements, and a 64-byte span.

A custom kernel read and wrote signed `int` values and computed `3*x + 7`.
All 32 results matched the host oracle exactly. The public version 2 contract
represents this precision and the Python frontend maps the unambiguous group
automatically. The legacy precision-bit contract remains FP16/FP32-only.

This establishes this same-precision unary carrier on NPU3720/compiler 8.3.
It does not by itself establish binary I32 carriers, mixed integer operations,
other integer widths, other graph shapes, or signed-overflow behavior. The
separate binary contract follows.

### Equal-span binary I32

A static two-input I32 `[1,32]` `Minimum` carrier produced one ACT group with
four invocations. Each parameter block contained input A at `+0x00`, input B
at `+0x28`, and the output at `+0x50`. All three records used element-type 9,
an eight-element count, a 32-byte span, and the established static dense CMX
layout; the output was disjoint from both inputs.

The compiler-produced `Minimum` graph matched its host oracle. A substituted
custom kernel then used signed `int` pointers and computed
`3*a - 2*b + 7`. The two independent inputs discriminated their roles and kept
all arithmetic in signed-I32 range. All 32 custom outputs matched an I64-widened
NumPy oracle exactly.

This establishes the equal-count, equal-span two-input I32 descriptor layout
for one carrier. It does not establish scalar/broadcast input records, unequal
shapes, additional arities, mixed integer precision, or signed
overflow semantics.

### Same-precision unary I8

A static I8 `[1,32]` `Abs` carrier has the same one-group/four-invocation
structure. Each input and output descriptor used element-type value 11, an
8-bit first stride, 16 elements, and a 16-byte span. The remaining static,
dense, CMX, and disjoint-output fields matched the supported unary structure.

A custom kernel read and wrote `signed char` values and computed `2*x + 3`
over an input domain chosen to remain within I8 range. All 32 results matched
the host oracle exactly. Version 2 represents I8 and Python maps this
unambiguous group automatically; the legacy precision flags remain unchanged.

This establishes one same-precision unary I8 carrier. It does not establish
binary I8 by itself, same-precision U8 custom kernels, saturation or
quantization semantics, overflow behavior, or other shapes. The separate
binary contract follows.

### Equal-span binary I8

A static two-input I8 `[1,32]` `Subtract` carrier produced one ACT group with
four invocations. Each parameter block contained input A at `+0x00`, input B
at `+0x28`, and the output at `+0x50`. All three records used element-type 11,
an eight-element count, an eight-byte span, and the established static dense
CMX layout; the output was disjoint from both inputs.

The compiler-produced `Subtract` graph matched its host oracle. A substituted
custom kernel then used `signed char` pointers and computed
`3*a - 2*b + 7`. The two independent inputs discriminated their roles and kept
every result in signed-I8 range. All 32 custom outputs matched an I16-widened
NumPy oracle exactly.

This establishes the equal-count, equal-span two-input I8 descriptor layout
for one carrier. It does not establish scalar/broadcast input records, unequal
shapes, additional arities, same-precision U8 custom execution, saturation,
quantization, or signed-overflow semantics.

### Same-precision unary I16

A static I16 `[1,32]` `Abs` carrier also produced one ACT group with four
invocations. Each input/output descriptor used raw element-type value 10, a
16-bit first stride, 16 elements, and a 32-byte span. Static, dense, CMX, and
disjoint-output fields matched the supported unary structure.

A custom kernel read and wrote signed `short` values and computed `3*x - 11`
over a bounded input domain. All 32 results matched the widened NumPy oracle
exactly. Version 2 represents I16 and Python maps this unambiguous group
automatically; the legacy precision flags remain FP16/FP32-only.

This establishes one same-precision unary I16 carrier on NPU3720/compiler 8.3.
It does not by itself establish binary I16, U16 custom kernels, saturation,
overflow behavior, mixed integer/float contracts, or other shapes. The
separate binary I16 and U16 contracts follow.

### Equal-span binary I16

A static two-input I16 `[1,32]` `Maximum` carrier produced one ACT group with
four invocations. Each parameter block contained input A at `+0x00`, input B
at `+0x28`, and the output at `+0x50`. All three records used element-type 10,
an eight-element count, a 16-byte span, and the established static dense CMX
layout; the output was disjoint from both inputs.

A custom kernel used signed `short` pointers and computed
`3*a - 2*b + 7`. The two independent host inputs were selected so swapping or
ignoring either role changes the expected output and all arithmetic remains in
signed-I16 range. All 32 outputs matched a widened NumPy oracle exactly.

This establishes the equal-count, equal-span two-input I16 descriptor layout
for one carrier. It does not establish scalar/broadcast input records, unequal
shapes, additional arities, saturation, or overflow
semantics. The compiler-produced `Maximum` result itself was incorrect on the
tested driver; the claim is limited to the substituted custom kernel.

### Equal-span binary U16

A static two-input U16 `[1,32]` `Subtract` carrier produced one ACT group with
four invocations. Each parameter block contained input A at `+0x00`, input B
at `+0x28`, and output at `+0x50`. All three records used raw element-type 6,
an eight-element count, a 16-byte span, and the established static dense CMX
layout; the output was disjoint from both inputs.

The compiler-produced `Subtract`, `Add`, and `Multiply` candidates matched
their U16 host oracles exactly. `Maximum` and `Minimum` returned corrupt
baseline values and were rejected. A substituted kernel used `unsigned short`
pointers and computed `(a >> 1) + (b >> 2) + 3`, with widened unsigned
intermediates and a domain that kept every result in range. All 32 outputs
matched NumPy exactly through copied and explicitly shared buffers.

By itself, this establishes one equal-count, equal-span two-input U16 carrier
on NPU3720/compiler 8.3. The separate scalar/broadcast and U32 contracts
follow; unary, additional-arity, saturation, overflow, and mixed precision
remain outside this result.

### Equal-span binary U32

A static two-input U32 `[1,32]` `Subtract` carrier produced one ACT group with
four invocations. Each parameter block used input A at `+0x00`, input B at
`+0x28`, and output at `+0x50`. All records used raw element type 5, an
eight-element count, a 32-byte span, and the established static dense CMX
layout; the output was disjoint from both inputs.

The compiler-produced `Subtract`, `Add`, and `Multiply` candidates matched
their U32 host oracles exactly. `Maximum` and `Minimum` returned corrupt
baseline data and were rejected. A substituted kernel used `unsigned int`
pointers and computed `(a >> 2) + (b >> 3) + 5`; all 32 outputs matched NumPy
through copied and explicitly shared buffers.

By itself, this establishes one equal-count, equal-span two-input U32 carrier
on NPU3720/compiler 8.3. The separate scalar/broadcast contracts follow;
unary, additional arity, saturation, overflow, mixed precision, and other U32
carrier operations remain outside this result.

### Unsigned scalar and row broadcasts

Static U16 and U32 `Subtract` graphs with `[1,32]` data and an independently
bound `[1,1]` input each produced one group with four invocations. The first
input and output contain eight elements per invocation. The second descriptor
contains exactly one element and one precision-sized span: two bytes for U16
or four bytes for U32. Custom kernels read this descriptor only at index zero.

For the exact source-level `[4,8]` plus `[1,8]` form, graph compiler 8.3
instead expands the row into an eight-element descriptor for every invocation.
The resulting `(8,8)->8` descriptor layout is structurally the same as the
equal-span carrier, and custom kernels index both inputs by the local element
index. Correct built-in and substituted `Subtract` execution established the
source-level broadcast meaning; descriptor inspection alone cannot recover it.

Both layouts were executed with values above the signed range for their
widths. Chained scalar and row custom kernels matched all 32 U16 and U32
outputs through copied and shared buffers.

A later non-commutative `[1,1]` minus `[1,32]` probe preserved the scalar as
input record zero and the data as record one. Each FP16, FP32, U16, and U32
carrier exposed four `(1,8)->8` invocations. Custom kernels using output-record
dimensions matched all copied and shared outputs. Together these probes
establish one graph-global scalar in either position.

The exact `[4,8]` with `[4,1]` per-row form produces the same four one-scalar
descriptors: `(8,1)->8` or `(1,8)->8`, depending on operand order. Distinct
values in all four rows reached their corresponding invocation, and custom
FP16, FP32, U16, and U32 kernels matched copied and shared host oracles. The
native descriptor alone does not say whether its scalar is graph-global or
row-specific; that meaning comes from the caller's source graph and validated
positional mapping. This does not establish multiple scalars, other shapes or
axes, signed integer scalar inputs, or general broadcasting within this binary
carrier family.

### Mixed-precision conversion

A connected static `[1,32]` graph produced a unary conversion record with an
FP32 input at `+0x00` and FP16 output at `+0x28`. Both descriptors had 32
elements and dense CMX layout, while their byte spans were respectively 128
and 64. A custom kernel reading `float` and writing `__fp16` replaced one
conversion range and matched a discriminating host reference at all 32 output
positions across two input sets.

This confirms that the descriptor sequence can carry different input/output
types and spans. A reverse connected graph was also tested with a 32-element
FP16 input (64 bytes) and FP32 output (128 bytes). A custom `__fp16`-to-`float`
kernel matched all 32 outputs across two input sets.

An independent static U8-to-FP32 `[1,32]` `Convert` graph produced one ACT
group with two full-tensor replicas. Each invocation described a 32-element,
32-byte U8 input using raw type 7 and a 32-element, 128-byte FP32 output. A
custom kernel read `unsigned char`, wrote `float`, and computed
`0.5*x - 7.25`; all 32 NPU outputs matched the host oracle exactly.

These three conversion paths establish a narrow unary descriptor convention.
They do not establish arbitrary mixed types or mixed-dtype binary operations.
The matching conversion ranges are executable tile-local replicas, not output
partitions; there is still no safe automatic mapping for a conversion group.
The version 2 public patch target represents the records, while Python requires
explicit prepared-group binding.

### Experimental mixed-dtype two-input invocation

A static `[1,32]` `GatherElements` carrier produced this descriptor sequence:

| Parameter offset | Role | Type | Count | Bit stride | Byte span |
| ---: | --- | --- | ---: | ---: | ---: |
| `+0x00` | data input | FP32 (`1`) | 32 | 32 | 128 |
| `+0x28` | indices input | I32 (`9`) | 32 | 32 | 128 |
| `+0x50` | output | FP32 (`1`) | 32 | 32 | 128 |

All three records were static dense CMX tensors. The compiler emitted two
full-tensor invocations: range 0 on tile 0 and range 1 on tile 1, with their
data addends separated by the same observed `0x200000` tile delta.

A custom kernel read the first record as `float`, the second as signed
32-bit indices, and wrote the third as `float`. Each index selected two
different positions from the data span, making both input records necessary
to match the oracle. The kernel matched all 32 outputs exactly for linear and
irregular data on tile 0. Rerouting the output DMA to tile 1 and replacing only
range 1 produced the same exact results there.

This confirms one mixed FP32/I32 two-input ACT convention. It does not establish
arbitrary integer tensors, negative or out-of-range indices, integer graph
inputs, broadcasting, other shapes, or other operators. The original carrier
implementation did not match a `GatherElements` host oracle in this
configuration, so this is custom-entry ABI evidence rather than a claim that
the carrier operator itself works. The public patch target intentionally
rejects this mixed two-input I32 descriptor even though it accepts the separate
same-precision I32 contracts above.

### Five-input multi-scalar FP16 contract

An FP16 `FakeQuantize` carrier with `[1,32]` data and four independently bound
`[1,1]` inputs produced one ACT group with four invocations. Every invocation
uses six consecutive records:

| Parameter offset | Role | Count | Span |
| ---: | --- | ---: | ---: |
| `+0x00` | data input | 16 | 32 bytes |
| `+0x28` | scalar input 1 | 1 | 2 bytes |
| `+0x50` | scalar input 2 | 1 | 2 bytes |
| `+0x78` | scalar input 3 | 1 | 2 bytes |
| `+0xa0` | scalar input 4 | 1 | 2 bytes |
| `+0xc8` | output | 16 | 32 bytes |

The unmodified graph matched its `FakeQuantize` host oracle. A substituted
kernel then used all four scalar records at index zero and matched all 32
logical outputs through copied and shared buffers. This confirms only these
positions, counts, precision, source shapes, and compiler/device combination;
it does not establish arbitrary high arity or multi-scalar layouts.

### Experimental scalar-broadcast input

Static `[1,32]` `Maximum` carriers with host-provided `[1,1]` scalar inputs
were compiled in FP16 and FP32 accuracy mode. In both precisions, the compiler
emitted four invocations with this descriptor shape:

| Parameter offset | Role | Count | FP32 span | FP16 span |
| ---: | --- | ---: | ---: | ---: |
| `+0x00` | data input | 8 | 32 bytes | 16 bytes |
| `+0x28` | scalar input | 1 | 4 bytes | 2 bytes |
| `+0x50` | output | 8 | 32 bytes | 16 bytes |

The scalar was not expanded into an eight-element CMX tensor. Invocations 0
and 2 reused one scalar address on tile 0; invocations 1 and 3 reused the
corresponding address on tile 1. A custom kernel read `scalar[0]` and computed
`data[i] * 1.375 - scalar[0] * 0.625`. Replacing all four ranges matched every
host-oracle value exactly for linear and irregular inputs in both precisions.

Incremental replacement exposed the output partitions. Range order was
interleaved by tile: ranges 0, 1, 2, and 3 controlled output positions 0–7,
16–23, 8–15, and 24–31 respectively. This is distinct from the full-tensor
tile replicas observed in the conversion and mixed-dtype probes.

This confirms one scalar-broadcast descriptor and execution contract, not
general broadcasting. Other scalar positions, dimensions, shapes, operators,
broadcast axes, and compiler versions remain unverified. The unmodified
`Maximum` carrier produced invalid-looking output in this configuration, so
the carrier is evidence for the custom entry ABI rather than for built-in
operator semantics. The legacy public patch target cannot express per-record
counts in general; it represents this exact two-input case with
`PATCHBLOB_CONTRACT_INPUT_1_SCALAR`. The version-2 target records each tensor
and also represents the later scalar-first `Subtract` carriers.

### Descriptor-level row broadcast

Static `[4,8]` data plus an independently bound `[1,8]` row input produced
four ACT invocations in both FP16 and precision-preserved FP32. The source
graph shapes are unequal, but every invocation presents the same dense binary
record sequence:

| Parameter offset | Role | Count | FP32 span | FP16 span |
| ---: | --- | ---: | ---: | ---: |
| `+0x00` | data input | 8 | 32 bytes | 16 bytes |
| `+0x28` | expanded row input | 8 | 32 bytes | 16 bytes |
| `+0x50` | output | 8 | 32 bytes | 16 bytes |

Custom kernels therefore index both inputs with the invocation-local element
index. The FP16 `Maximum` and FP32 `Subtract` recipes matched copied and shared
host oracles. This confirms only the descriptor contract presented by these
carriers; it does not determine whether row storage is copied or reused, nor
does it establish other ranks, axes, shapes, or general broadcasting.

## How `patchblob` substitutes code

The supported mutation is deliberately narrow:

1. Validate the graph ELF, required sections, relocation tables, selected
   groups, and tensor contracts.
2. Validate the standalone kernel ELF and extract only its executable `.text`
   image.
3. Append that image to `.text.KernelText` at an aligned offset.
4. Round the inserted extent so rebased sections retain their declared ELF
   alignment, then update their file offsets.
5. Update the selected range extents and their `KernelText` relocation
   addends.
6. Verify that invocation records, parameters, tensor metadata, runtime
   configuration, DMA/barrier/DPU scheduling data, and unrelated relocations
   did not change.

The tested construction uses 0x400-byte image alignment and at least 0x80 bytes
of zero tail padding. Additional zero padding may be added to preserve the
alignment of every shifted section. These are conservative working values,
not proven hardware minima. Several selected ranges may point to the same
appended image.

Passing ELF validation is not enough to prove that arbitrary code follows the
ACT entry convention. `npunlock` therefore also requires a compatible carrier
contract and treats host-side semantic comparison as the correctness oracle.

## Creating the graph through Level Zero

`ir2blob` asks the Intel NPU driver to compile OpenVINO-format IR without
linking or loading the OpenVINO runtime. The confirmed graph-extension path is
`pfnCreate2` with an in-memory `ZE_GRAPH_FORMAT_NGRAPH_LITE` payload:

```text
u16 compiler_major
u16 compiler_minor
u32 payload_count = 2
u64 xml_size
u8  xml[xml_size]
u64 weights_size
u8  weights[weights_size]
```

The compiler version is queried from the driver. `pBuildFlags` must be a
non-null C string even when it is empty. After compilation, `ir2blob` copies
the complete native ELF using `pfnGetNativeBinary2` before releasing the graph.

`pfnCreate3` is not part of the supported path.

## Unsupported assumptions

Do not infer support for:

- dynamic shapes or arbitrary strides/layouts;
- supported patch targets beyond the narrow same-precision FP16, unary FP32,
  unary I32/I16/I8, equal-span binary FP32/I32/I16/I8/U16/U32, and
  the scalar/row-broadcast cases above (mixed FP16/FP32 and U8-to-FP32
  conversion remain limited to their explicit version 2 contracts);
- general broadcasting or unequal binary input shapes beyond the experimental
  one-element scalar and `(4,8)` plus `(1,8)` row cases above;
- arbitrary compiler or graph-format versions;
- nonempty `KernelData` for custom kernels;
- unresolved data/code fixups or helper runtimes;
- high pointer halves;
- cross-invocation neighborhoods or graph-global coordinates;
- arbitrary ACT record counts; or
- mapping a source node name directly to an ACT range.

A graph with no compatible ACT carrier, including a DPU-only graph, must be
rejected rather than patched speculatively.

## Related public documentation

- [KERNEL_ELF.md](KERNEL_ELF.md) describes the standalone SHAVE kernel ELF.
- [MOVITOOLS.md](MOVITOOLS.md) describes the compiler DLL invocation boundary.
- [C API](../docs/C_API.md) documents the supported public library calls.
- [README](../README.md) gives an end-to-end overview.
