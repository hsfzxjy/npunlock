# NPU3720 native graph ELF: observed ABI

Status: experimental. Last updated 2026-09-21.

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
| section-header size | 64 bytes |

A parser therefore cannot apply ordinary x86-64 ELF identity checks. It must
validate the observed NPU container fields and then bounds-check every section
and relocation it uses.

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

The exact section set, order, size, and file offsets vary by graph. Absolute
file offsets are not an ABI. `npunlock` follows section references and
relocations instead.

Observed relocation tables use 24-byte ELF64 `SHT_RELA` entries. Relocation
type 4 is used for ACT component pointers. Its official vendor name and write
width are unknown, so the project supports only the exact uses described here.
Some loader-managed tables use nonstandard section links; they are outside the
patching contract.

## ACT invocation records

For the supported compiler family, an `ActKernelInvocation` record is 0x40
bytes.

| Record offset | Observed purpose | Confidence |
| ---: | --- | --- |
| `+0x00` | index of the selected `ActKernelRange` | confirmed |
| `+0x04` | relocated reference to this invocation's `KernelParams` | confirmed |
| `+0x08` | relocated reference to `KernelData` | confirmed; supported custom kernels use no data |
| `+0x34` | tile-like value (`0` or `1` in two-tile graphs) | inferred |

The number of invocation records is not the number of graph operations. It
depends on shape, layout, compiler partitioning, and tile configuration. For
example, some `[1,16]` FP16 operations have two invocations of eight elements
even when compilation requests one NPU tile.

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

1. every computational node has one validated positional ACT group;
2. node count and group count match exactly in topological order;
3. the selected group's input arity matches the custom node; and
4. every invocation in the group has the same supported tensor contract.

Graphs containing DPU operations, fused nodes, inserted conversions, or other
count mismatches need an explicit validated target selection. A source node
name is not stored as a reliable range selector in the native blob.

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
| `+0x18` | element-type value: 2 for FP16, 1 for validated FP32 |
| `+0x24` | raw value 2 for supported CMX tensors |

The official enum names for the raw values are not known. `npunlock` validates
the complete combination instead of assigning broader meanings to individual
numbers.

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

An FP16 span is `element_count * 2`; an FP32 span is
`element_count * 4`. Inputs and output must describe matching element counts,
and the output span must not alias an input span.

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

## How `patchblob` substitutes code

The supported mutation is deliberately narrow:

1. Validate the graph ELF, required sections, relocation tables, selected
   groups, and tensor contracts.
2. Validate the standalone kernel ELF and extract only its executable `.text`
   image.
3. Append that image to `.text.KernelText` at an aligned offset.
4. Rebase section file offsets affected by the insertion.
5. Update the selected range extents and their `KernelText` relocation
   addends.
6. Verify that invocation records, parameters, tensor metadata, runtime
   configuration, DMA/barrier/DPU scheduling data, and unrelated relocations
   did not change.

The tested construction uses 0x400-byte image alignment and 0x80 bytes of tail
padding. These are conservative working values, not proven hardware minima.
Several selected ranges may point to the same appended image.

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
- element types other than the narrow FP16 and unary FP32 cases above;
- broadcasting or unequal binary input shapes;
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
