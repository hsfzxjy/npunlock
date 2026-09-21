# Observed 3720xx ACT kernel ELF ABI

Status: experimental, last reconciled with retained execution evidence on
2026-09-21.

This document describes the standalone linked kernel accepted by the current
`npunlock` MVP and the effective entry contract exercised by custom C kernels.
It is not a published Movidius/Intel ABI. “Confirmed,” “strong inference,” and
“open” have the meanings defined in `GRAPH_ELF.md`.

## Linked ELF container

The accepted standalone result is:

| Property | Confirmed value |
| --- | --- |
| class | ELF32 |
| byte order | little-endian |
| type | `ET_EXEC` (2) |
| machine | SPARC value 2, used here for SHAVE |
| flags | 0 in retained outputs |
| entry address | `0x1d000000` |
| executable image | `.text` at `0x1d000000` |
| data image | `.arg.data` at `0x1e000000`, empty in supported kernels |

The executable `PT_LOAD` covers `.text`; the retained outputs use 16-byte
alignment. The linked result must have bounded section/program extents, a
unique executable image, no undefined symbols, and no remaining relocation
sections. An entry symbol is retained at the start of `.text` in the tested
builds.

The linker script used by the successful experiments performs this grouping:

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

Consequently the graph-embeddable code image is the linked `.text` bytes,
including linked read-only data and linker padding that fall in that output
section. It is not the entire ELF file. For the upstream Abs and Exp reference
ELFs, the complete `.text` section matched the selected graph `KernelText`
slice byte-for-byte and its size matched the range extent.

The current supported path requires `.arg.data` to be empty. Placement of
nonempty kernel data, runtime archives, unresolved fixups, or multiple loadable
images is open and must not be approximated by concatenating sections.

## Build recipe proven for the MVP

The successful in-memory three-stage recipe is:

```text
moviCompile64!main
  moviCompile.dll -cc1 -triple shave -target-cpu 3720xx
  -S -O2 -x c entry.c -ffunction-sections -fdata-sections -o -

moviAsm64!process
  moviAsm.dll --cv 3720xx --noSPrefixing

moviLLD64!process
  moviLLD.dll -flavor gnu -EL -e <entry> -z max-page-size=0x10
  inputs: assembled ELF object, then shave_kernel.ld
```

`shavecc` embeds the proven Apache-2.0 `shave_kernel.ld` bytes as its default
second linker input. A caller-supplied non-empty script view replaces those
bytes; this changes configuration, not the observed linker ABI above.

Compiler definitions used by the generalized local-window sources were
restricted to explicit uppercase `NAME=DECIMAL` arguments. No runtime archive
was needed for the confirmed add-one, local-neighbor, or two-input kernels.

## Effective ACT entry contract

The source-level entry tested by execution is:

```c
void controlled_act(unsigned layerParams);
```

Generated code receives the argument in integer register `i18` and returns
through `i30`. Ordinary compiled returns worked for all confirmed custom
kernels. This is an effective calling convention for the calibrated carrier,
not a complete vendor declaration.

`layerParams` is a usable 32-bit address of the invocation-selected
`KernelParams` block. The successful C kernels used 32-bit little-endian loads
for pointer-bearing fields. High address bits were not tested.

For the observed 0x28-byte tensor descriptor:

```text
record +0x00  low 32 bits of data address
record +0x0c  u32 rank
record +0x10  low 32 bits of dimensions address
```

The dimensions are `rank` little-endian four-byte values. Successful kernels
checked rank 1 through 15, multiplied dimensions with an explicit capacity
guard, and treated the buffers as contiguous FP16 arrays. Host-side preflight
separately validated the complete static, dense, FP16, CMX, disjoint-output
contract, including strides and relocations; the kernel source itself did not
implement general layout handling.

Confirmed parameter records relative to `layerParams`:

```text
unary:  input +0x00, output +0x28
binary: input A +0x00, input B +0x28, output +0x50
```

The binary inputs were internal ACT tensors created by earlier stages of a
graph with one host input. Two independently bound host inputs remain open.

## Confirmed execution envelope

Source-built kernels established all of the following on NPU3720:

- one input plus FP16 add-one over invocation-local chunks of 8 and 16
  elements;
- one compiled image selected by more than one range;
- indexed local reads and nonlinear FP32 arithmetic with FP16 output;
- an eight-byte generated stack save/restore frame;
- invocation-local counts of 64, 128, and 2048 FP16 elements;
- reads from index 0 through 2047 inside a 4096-byte advertised input span;
- two tensor operands, including a local neighbor read from the second input.

The nonlinear kernel used separate FP32 multiply/add operations, sign-based
selection, and output conversion to FP16. The observed small stack frame
worked, but stack capacity, calls, recursion, callee-save rules, and helper
runtime behavior are open.

The graph scheduler owns chunking. A custom kernel sees the current
invocation's descriptor and buffers, not an automatic graph-global tensor
view. Neighbor kernels replicated invocation-local endpoints. Cross-chunk
halos and global coordinates were not established.

## Representative artifacts

The simplest semantic kernel, `add1-fp16`, has:

| Property | Value |
| --- | --- |
| complete ELF size | 828 bytes |
| `.text` size | `0xc0` (192 bytes) |
| entry-symbol extent | 187 bytes |
| ELF SHA-256 | `3f9d52273870c2e911000da3c3bd474c0514bd297d29539a14e935b2b01de6d5` |
| `.text` SHA-256 | `bfe23abd677c5594c2956d1912f6d87aa87c62a6119bd8c85ab2a5fa08aaffab` |

A fresh three-stage build reproduced both hashes byte-for-byte.

The nonlinear neighbor kernel has a 400-byte (`0x190`) `.text` image and a
1036-byte ELF. Its ELF SHA-256 is
`2b50e7114f1243ec653e1afd60981c25acd29f6c5543ab25234e185ba367a186`;
its `.text` SHA-256 is
`9232118baf5f8a8e72f9d9856c776cd3ed2c416f9fcd776ff18b40b531f99990`.
A repeat build reproduced both.

Seven later variants covered 0xc0, 0x110, 0x120, and 0x130 text sizes. They
were all ELF32 little-endian SPARC executables with entry/text at
`0x1d000000`, empty `.arg.data`, no undefined symbols, and no remaining
relocations. Those seven did not receive separate repeat-build comparisons.

## Loader mapping into a graph

The graph does not consume this ELF container directly. The supported bridge
is:

```text
validated kernel ELF .text bytes
  -> aligned bytes inside graph .text.KernelText
  -> ActKernelRange code relocation addend selects the image base
  -> ActKernelRange +0x0c records the image extent
```

The standalone virtual address remains `0x1d000000`; the graph range record
also carries this observed code address. A single appended image may serve
multiple ranges. The tested graph append alignment and tail padding are
`0x400` and `0x80`, respectively, but neither is established as a minimum.

## Validation boundary

Before accepting a linked kernel, the MVP validates at least:

- ELF magic, class, endianness, version, type, machine, and entry;
- section table, program table, string table, and symbol table bounds;
- `.text` address, flags, nonempty extent, and executable-load coverage;
- `.arg.data` address and zero size;
- absence of undefined symbols and remaining relocation sections;
- absence of integer overflow in every derived range.

That structural validation does not prove compatibility with an arbitrary
carrier. The `dummy.3720xx.elf` negative control was structurally appendable
and graph-loadable but lost the device after submission. Explicit carrier
preflight and a host semantic oracle remain necessary.

## Open ABI surface

No supported claim is made for nonempty `.arg.data`, global/static mutable
data, helper libraries, unresolved or dynamic relocations, multiple code
images, alternate entry addresses, high pointer bits, dynamic dimensions,
non-dense strides, layouts other than the calibrated carriers, other dtypes,
arbitrary arity, graph-global neighborhoods, larger stack use, exceptions, C++
runtime behavior, or other SHAVE targets.

## Primary evidence

```text
D:\srcs\level-zero\samples\npurun\NPU_ABI_STATUS.md (sections 14-22)
D:\srcs\level-zero\samples\npurun\experiments\movitools-custom-kernel\
  artifacts\reference-kernels\README.md
  oem-tool-invocation\README.md
  act-entry-trial\README.md
  act-entry-trial\add1-fp16.c
  act-entry-trial\builds\add1-fp16\elf.json
  shape-portability\README.md
  shared-complex\README.md
  local-window-multi\README.md
```
