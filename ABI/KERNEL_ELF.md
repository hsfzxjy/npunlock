# NPU3720 ACT kernel ELF: observed ABI

Status: experimental. Last updated 2026-09-21.

This document describes the standalone SHAVE executable produced by
`shavecc`, and the C entry convention used when that code runs as an ACT
kernel inside an NPU3720 graph.

It is an observed compatibility contract, not a published Intel or Movidius
ABI. Read [GRAPH_ELF.md](GRAPH_ELF.md) first for the difference between the
standalone kernel ELF and the native graph ELF that contains it.

## The two ELF files are different

`npunlock` works with two ELF formats:

```text
standalone kernel ELF (ELF32)
  contains the custom SHAVE executable
             |
             | extract validated .text bytes
             v
native graph ELF (ELF64)
  contains scheduling, tensor metadata, and KernelText
```

The graph never receives the complete standalone ELF. `patchblob` extracts
the executable `.text` image and inserts only those bytes into the graph's
`.text.KernelText` section.

## Required standalone ELF structure

The supported linked result has these properties:

| Property | Required observed value |
| --- | --- |
| ELF class | ELF32 |
| byte order | little-endian |
| type | `ET_EXEC` (2) |
| machine | 2 (SPARC value used by these SHAVE tools) |
| entry address | `0x1d000000` |
| executable section | `.text` at `0x1d000000` |
| data section | `.arg.data` at `0x1e000000`, empty |

An executable `PT_LOAD` segment must cover `.text`. Current outputs use
16-byte alignment. The file must contain one unambiguous executable image,
bounded section and program tables, no undefined symbols, and no remaining
relocation sections.

The machine value does not mean the kernel runs on a conventional SPARC CPU.
It is the ELF identity emitted by the MoviTools SHAVE toolchain used here.

## Linker layout

The default linker script places data and code at the addresses expected by
the validated graph carriers:

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

Read-only constants that the linker places in the output `.text` section are
part of the extracted code image. Linker padding inside that section is also
preserved.

The current contract requires `.arg.data` to be empty. Nonempty mutable data
would need graph placement and relocation behavior that has not been
validated. It must not be approximated by concatenating ELF sections.

The Apache-2.0 default script is stored at
[`src/shavecc/shave_kernel.ld`](../src/shavecc/shave_kernel.ld) and embedded in
the library at build time. A caller may explicitly supply a replacement, but
the returned ELF must still pass the same validation.

## Build pipeline

`shavecc` runs three in-memory stages:

```text
C source bytes
  -> moviCompile64.dll
SHAVE assembly bytes
  -> moviAsm64.dll
relocatable ELF32 object
  -> moviLLD64.dll + linker script
linked executable ELF32
```

The confirmed target selector is `3720xx`. The effective arguments are:

```text
moviCompile.dll -cc1 -triple shave -target-cpu 3720xx
  -S -O2 -x c entry.c -ffunction-sections -fdata-sections -o -

moviAsm.dll --cv 3720xx --noSPrefixing

moviLLD.dll -flavor gnu -EL -e <entry> -z max-page-size=0x10
  --gc-sections <MoviTools-root>\lib\mlibm.a
```

The source, assembly, object, linker script, and result stay in memory. The
nominal `entry.c` name is a compiler argument, not an intermediate file.

`mlibm.a` supplies math functions such as `tanhf`. Section garbage collection
is important: without `--gc-sections`, unrelated archive members can introduce
writable global data in `.arg.data`, making the kernel unsupported. With
garbage collection, the validated GELU examples retain only the reachable math
closure and still produce an empty data section.

Simple arithmetic kernels do not require a runtime archive beyond this
linker's normal inputs.

## C entry point

The execution-tested source signature is:

```c
void controlled_act(unsigned layerParams);
```

The compiler passes `layerParams` in integer register `i18`; ordinary compiled
returns use `i30`. This describes the entry behavior needed by the current
carriers, not a complete SHAVE calling-convention specification.

`layerParams` is the low 32-bit address of the parameter block selected for
the current graph invocation. Successful kernels read pointer-bearing fields
with explicit little-endian 32-bit loads. High address bits have not been
validated.

Kernel source may start with the virtual include
`#include <npunlock/npu3720_kernel.h>`. `shavecc` expands it from an embedded
copy before compilation. Its target-scoped helpers implement the descriptor
loads and guarded element-count calculation described below; they are a
convenience over this observed contract, not evidence of a broader ABI.

## Reading tensor descriptors

The parameter block starts with one or more 0x28-byte tensor descriptors. A
kernel normally needs these fields:

```text
descriptor +0x00  low 32 bits of tensor data address
descriptor +0x0c  rank as u32
descriptor +0x10  low 32 bits of dimensions-array address
```

The dimensions array contains `rank` little-endian 32-bit values. A custom
kernel should:

1. reject rank zero or a rank larger than its own bound;
2. reject zero dimensions;
3. check multiplication before calculating the element count; and
4. never process beyond the element count described for this invocation.

Host-side `patchblob` validation checks the fields that the small kernel does
not: static shape, dense bit strides, supported precision, CMX placement,
matching input/output spans, and non-aliasing output.

Validated descriptor positions are:

```text
unary kernel:
  input  at layerParams + 0x00
  output at layerParams + 0x28

binary kernel:
  input A at layerParams + 0x00
  input B at layerParams + 0x28
  output  at layerParams + 0x50
```

Both internal operands and two independently bound host inputs have been
validated for a narrow static dense FP16 binary carrier. This does not imply
broadcasting, unequal shapes, or arbitrary input counts.

## Invocation-local execution

The graph scheduler divides a tensor into invocation-local chunks. A custom
kernel receives only the descriptor for its current chunk; it does not
automatically receive a full graph-global tensor view.

Confirmed FP16 kernels include:

- elementwise add-one over chunks of 8 and 16 elements;
- one code image selected by several graph ranges;
- FP32 intermediate arithmetic with FP16 output;
- two-input arithmetic with independently bound host inputs;
- invocation-local chunks of 64, 128, and 2048 elements; and
- reads across the full advertised 4096-byte span of a 2048-element chunk.

The last result is a lower bound for local reach, not permission to cross an
invocation boundary. Neighbor-based kernels must define their boundary
behavior within each chunk.

A small generated stack frame has executed successfully. Larger stack use,
function calls, recursion, callee-save rules, and general runtime-library
behavior remain outside the supported contract.

## How the code image is installed

After kernel validation, `patchblob` performs this mapping:

```text
kernel ELF .text bytes
  -> aligned append inside graph .text.KernelText
  -> selected ActKernelRange relocation points to the append offset
  -> selected ActKernelRange extent records the image size
```

The standalone virtual address and supported graph range both use
`0x1d000000`. Several ranges may reference one appended image. The current
patcher uses 0x400-byte append alignment and 0x80 bytes of tail padding as
conservative, execution-tested values; they are not proven minima.

## Validation performed by `shavecc`

Before returning success, the library checks at least:

- ELF magic, class, byte order, type, machine, and entry address;
- section, program, string, and symbol table bounds;
- a nonempty executable `.text` at the expected address;
- executable-load coverage of `.text`;
- `.arg.data` at the expected address with size zero;
- absence of undefined symbols;
- absence of remaining relocation sections; and
- overflow safety for every derived offset and extent.

These checks establish structural compatibility, not computation correctness.
The selected graph carrier must also pass its tensor-contract checks, and an
application must compare output with a host oracle when validating new kernel
semantics.

## Unsupported assumptions

Do not infer support for:

- nonempty `.arg.data` or mutable global/static storage;
- arbitrary helper libraries or data-bearing library closures;
- unresolved, dynamic, or runtime relocations;
- multiple executable images or alternate entry addresses;
- high pointer halves;
- dynamic shapes, non-dense layouts, or arbitrary precisions;
- broadcasting, unequal binary shapes, or arbitrary arity;
- graph-global neighborhoods or cross-chunk halos;
- large stacks, exceptions, or a C++ runtime; or
- SHAVE targets other than `3720xx`.

## Related public documentation

- [GRAPH_ELF.md](GRAPH_ELF.md) describes the native graph container and range
  records.
- [MOVITOOLS.md](MOVITOOLS.md) describes the compiler DLL boundary.
- [C API](../docs/C_API.md) documents `shavecc` and `patchblob` ownership and
  error handling.
