# Observed NPU3720 native graph ELF ABI

Status: experimental, last reconciled with the retained research evidence on
2026-09-21.

This document records the narrow native-graph contract used by `npunlock`. It
is not an Intel specification and must not be treated as a general VPU ELF
ABI. The evidence comes from compiler-produced graphs on one Meteor Lake
system (device `0x7d1d`, graph extension 1.18, compiler 8.3) and controlled
mutations that were executed on that system.

The labels used below are deliberate:

- **Confirmed** means a byte-level observation or a controlled execution
  established the behavior.
- **Strong inference** means several observations support the interpretation,
  but the formal field definition is not known.
- **Open** means the research did not establish a contract.

## Container identity

**Confirmed:** native graph blobs are section-oriented ELF64 little-endian
containers. A representative FP16 `Abs(Add(...))` carrier has the following
header values:

| Field | Observed value |
| --- | ---: |
| ELF class | 2 (`ELFCLASS64`) |
| data encoding | 1 (little-endian) |
| `e_type` | 1 (`ET_REL` value) |
| `e_machine` | 0 |
| `e_ident[EI_VERSION]` / `e_version` | 0 / 0 |
| entry point | 0 |
| program headers | none |
| section-header size | 64 bytes |

The zero/nonstandard identity values are intentional observations. A parser
must not require an ordinary host-machine ELF identity merely because the
container otherwise follows ELF64 section-table conventions.

Representative sections include:

```text
.metadata
.text.actKernelRtConfigSec
.text.dmaTasks0
.text.dmaTasks1
.text.BarrierConfigs
.text.KernelText
.text.KernelData
.text.KernelParams
.text.ActKernelRanges
.text.ActKernelInvocations
.text.MappedInference
.text.DPUInvariants
.text.DPUVariants
.perf.metrics
.note.LoaderABIVersion
.note.MappedInferenceVersion
.meta.PlatformInfo
compatibility_string
.data.ConstIO / .data.BuffersIO
.symtab.*
.rlt.*
```

Names, offsets, sizes, record counts, and the presence of optional sections
vary with the graph. Section-relative references and relocations are the
contract used by the MVP; absolute file offsets are not.

Relocation sections observed by the current parser use `SHT_RELA` entries of
24 bytes. Ordinary section-symbol relocations use the standard ELF64-style
`r_offset`, packed symbol/type word, and signed addend. Relocation type 4 is
the observed pointer relocation used for ACT component references; its formal
vendor name and write width are not claimed here.
Some loader-managed relocations use a non-section `sh_link` such as `0xff20`;
their complete arithmetic and namespace are open.

## ACT object relationships

The confirmed navigation path is:

```text
ActKernelInvocation
  -> relocation-selected KernelParams base
  -> optional KernelData reference
  -> selected ActKernelRange
  -> relocation-selected slice of KernelText
```

These edges matter more than physical adjacency. A graph can contain support
kernels, multiple logical ACT operations, and multiple invocation records for
one logical operation.

### `ActKernelInvocation`

**Confirmed for the studied graph family:** records are 0x40 bytes and the
raw `u32` at record-relative `+0x00` is the selected range index. Changing
only this word redirected execution to another implementation.

| Offset | Observed use | Confidence |
| ---: | --- | --- |
| `+0x00` | zero-based `ActKernelRange` selection index | confirmed for tested family |
| `+0x04` | relocation to the invocation's `KernelParams` base | confirmed |
| `+0x08` | relocation to `KernelData` | confirmed relationship; data was empty in custom trials |
| `+0x34` | raw tile candidate (`0/1` in two-tile graphs) | strong inference |

Every tested invocation also had a special relocation at `+0x00` with type 5,
symbol 2, and addend `0x18`. The addend equals the observed range-record
stride. The range-index behavior is confirmed; the formal meaning of this
special relocation is open.

Invocation count is not an operation count or a simple tile multiple. Shape,
layout, compiler partitioning, and `NPU_TILES` all affected it. For example,
an FP16 `[1,16]` graph configured for one NPU tile still had two invocations,
each covering eight elements.

### Observed positional invocation groups

Controlled graph-compiler 8.3 comparisons of `[1,32]` unary chains
`Abs -> Abs`, `Exp -> Abs`, and `Abs -> Exp` established a narrow positional
grouping rule for that graph family. Each source operation produced four
contiguous invocation/range records. Bytes `+0x0c..+0x2f` and
`+0x3c..+0x3f` were invariant within one operation and changed at the source
operation boundary. The range index at `+0x00`, relocation slots at `+0x04`
and `+0x08`, and per-invocation/tile-like fields at `+0x30`, `+0x34`, and
`+0x38` are excluded from this identity.

The two adjacent `Abs` operations shared the same `KernelText` image but still
had different invocation identities. Therefore code relocation alone is not a
valid operation-group selector.

`patchblob` uses the invariant slices only to discover contiguous positional
groups, then validates every invocation's range relocation and complete tensor
contract. A high-level caller may correlate these groups with topologically
ordered source operations only when it independently knows the entire
computational graph is represented by those ACT groups and the counts and
arities agree. This observation does not establish arbitrary node-name
mapping, mixed ACT/DPU graph mapping, or stability across compiler families.

### `ActKernelRange`

**Confirmed for the studied graph family:** records are 0x18 bytes. The
fields used by substitution are:

| Offset | Observed use | Confidence |
| ---: | --- | --- |
| `+0x04` | code virtual address, `0x1d000000` | confirmed in compatible ranges |
| `+0x08` | code pointer relocated from `.text.KernelText` | confirmed |
| `+0x0c` | selected code-image extent in bytes | strong inference from layout; execution-proven when changed with the base |

The relocation addend at `+0x08` is the byte offset into `KernelText`. The
range extent bounds the selected image. Standalone Abs and Exp `.text` images
matched their selected graph slices byte-for-byte:

| Implementation | `KernelText` base | Extent | FNV-1a-64 |
| --- | ---: | ---: | --- |
| support/shared | `0x0000` | `0x61c0` | `8ae62928624d3d7c` |
| Exp | `0x6400` | `0x08a0` | `5cce99108e61b81d` |
| Abs | `0x7000` in a combined graph | `0x14a0` | `0e29aba2c3759fbc` |

The same image may be selected by multiple ranges. This is confirmed graph
metadata sharing; it does not prove physical instruction-cache or residency
sharing across tiles.

### `KernelParams` and the observed `MemRefData` record

An invocation's `+0x04` relocation selects a base inside
`.text.KernelParams`. In several compiler-generated ACT implementations,
successive bases were 0xc0 bytes apart. That is a confirmed cadence for those
implementations, not a universal parameter-block size.

The custom-kernel carriers expose consecutive 0x28-byte tensor descriptors.
The useful effective layout is:

| Offset | Raw/effective meaning |
| ---: | --- |
| `+0x00` | data pointer, supplied through a special relocation; custom code used its low 32 bits |
| `+0x08` | raw value 1 in the accepted static carriers |
| `+0x0c` | `u32` rank |
| `+0x10` | relocation-selected pointer to `rank` signed 32-bit dimensions |
| `+0x14` | relocation-selected pointer to `rank` signed 64-bit bit-strides |
| `+0x18` | raw value 2 in FP16 carriers and 1 in the validated FP32 carrier |
| `+0x24` | raw value 2 in the accepted CMX carriers |

The formal enum names of the raw values at `+0x08`, `+0x18`, and `+0x24` are
not claimed. `npunlock` treats the complete observed combinations as narrow
static FP16/FP32 CMX contracts.

For dense FP16 validation, non-singleton dimensions sorted by increasing
stride must have bit strides `16`, then `16 * prior_dimension`, and so on.
Singleton-dimension strides differed between valid compiler records and are
not used to reject density. The tensor element count is the product of the
positive dimensions and its byte span is `count * 2`.

**Confirmed on compiler 8.3:** a plain FP32 `[1,2048]` Abs carrier is lowered
to three ACT groups: FP32-to-FP16 conversion, four FP16 Abs invocations, and
FP16-to-FP32 conversion. Patching the middle group therefore cannot preserve
FP32 precision. Serializing `DisablePrecisionConversion` with value
`dynamic:f16` on the Abs layer and compiling with
`EXECUTION_MODE_HINT="ACCURACY"` instead produces one four-invocation FP32 ACT
group. Each invocation advertises 1024 elements, 32-bit initial dense stride,
and a 4096-byte input/output span. A replacement FP32 GELU kernel executed
through that carrier with maximum absolute error `2.38419e-07` against the
host FP32 reference. This is a validated unary carrier observation, not a
general FP32 graph contract.

Confirmed descriptor roles:

- unary carrier: input at parameter base `+0x00`, output at `+0x28`;
- tested binary carrier: input A at `+0x00`, input B at `+0x28`, output at
  `+0x50`.

The binary observation is for internal ACT operands produced from one
host-bound graph input. It does not establish independently bound host inputs,
broadcasting, unequal shapes, or arbitrary descriptor counts.

Observed `[1,16]`, one-tile records used rank 4, dimensions `[8,1,1,1]`, bit
strides `[16,256,256,256]`, and raw order `0x2431`. Larger carriers advertised
64, 128, and 2048 FP16 elements per invocation. Custom code successfully read
indices 4094 bytes apart within the largest advertised 4096-byte input span.
This is a lower bound, not a maximum and not permission to cross an advertised
invocation boundary.

Kernel-specific scalar data may follow the descriptors. For one compiler
Clamp implementation, lower and upper FP32 values were at parameter-base
`+0x50` and `+0x54`. That layout must not be applied to unrelated kernels.

## Confirmed code-substitution mutation

The supported mutation is intentionally small:

1. Validate the graph ELF, required sections, relocations, explicit or
   positionally discovered invocation/range indices, and the expected tensor
   contract.
2. Extract the linked kernel ELF's executable `.text` image; never append the
   complete ELF container.
3. Grow `.text.KernelText`, preserving its original bytes as a prefix.
4. Rebase `e_shoff` and every later nonzero section file offset by the inserted
   byte count; update only the `KernelText` section size.
5. For each selected range, write the image byte size at range `+0x0c` and the
   shared image base into the matching `KernelText` relocation addend.
6. Preserve invocation records, parameters, `KernelData`, runtime config,
   tensor metadata, DMA/barrier/DPU scheduling content, and unrelated
   relocation contents.

Alignment `0x400` and tail padding `0x80` are execution-tested construction
choices. They are not proven minimum requirements. Multiple selected ranges
successfully referenced one appended image.

An appended known-compatible Abs image and several source-built kernels
executed successfully. An appended standalone `dummy` image parsed and loaded
but caused device loss after submission. Therefore graph structural validity
and shape-compatible metadata do not prove entry-ABI compatibility.

## Native graph creation and export context

Two Level Zero graph-extension uses were confirmed:

- load an exported graph ELF as the native/precompiled format;
- compile an in-memory `ZE_GRAPH_FORMAT_NGRAPH_LITE` payload through
  `pfnCreate2`, then copy the complete native ELF with
  `pfnGetNativeBinary2` before graph destruction.

The tested `NGRAPH_LITE` `ALL_WEIGHTS_COPY` payload is little-endian:

```text
u16 compiler_major
u16 compiler_minor
u32 payload_count = 2
u64 xml_size
u8  xml[xml_size]
u64 weights_size
u8  weights[weights_size]
```

The compiler version is queried from the driver. On the tested driver,
`pBuildFlags` must point to a non-null empty C string when no flags are used.
An initial `pfnCreate3` attempt terminated; `pfnCreate2` is the confirmed path.
This wire payload consumes OpenVINO-format IR, but using it does not require
linking or loading OpenVINO.

## Explicitly unsupported conclusions

The research does not establish a stable ABI for dynamic shapes, other data
types, arbitrary strides/layouts, high pointer halves, arbitrary ACT record
counts, arbitrary graph versions, nonempty kernel data, data/code fixups,
helper runtimes, cross-invocation halos, or mapping a source node name to an
ACT range. DPU-only graphs have no compatible ACT carrier.

## Primary evidence

The retained research tree is the evidence source; it is not vendored into
`npunlock`:

```text
D:\srcs\level-zero\samples\npurun\NPU_ABI_STATUS.md
D:\srcs\level-zero\samples\npurun\artifacts\component-contract\
D:\srcs\level-zero\samples\npurun\experiments\movitools-custom-kernel\
  artifacts\reference-kernels\README.md
  artifacts\mutated\README.md
  act-entry-trial\README.md
  shape-portability\README.md
  shared-complex\README.md
  local-window-multi\README.md
  direct-ir\README.md
```
