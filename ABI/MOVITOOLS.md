# Observed MoviTools DLL ABI and invocation contract

Status: experimental, last reconciled with the real-DLL integration result on
2026-09-21.

This document records the effective Windows x64 interfaces used to compile,
assemble, and link the confirmed `3720xx` ACT kernels. These exports are OEM
tool interfaces, not a supported public SDK ABI. Exact unobserved native types,
large-buffer behavior, reentrancy, and forward compatibility remain unknown.

## Provenance

The successful tool set was selected explicitly by the caller from:

```text
D:\Drivers\NPU\MVC_DEPEND\bin
```

That path is machine-local evidence, not a project default. `npunlock` does
not search arbitrary driver directories and must not package, modify, or
redistribute these binaries.

| DLL | Version observed | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| `moviCompile64.dll` | `00.114.11.4207` | 65,488,136 | `f5e164f0d9e02574ff94676f6bb799597640602ccdbb685724f241a8ed170c2e` |
| `moviAsm64.dll` | `1.13.16 64-bit` | 7,488,264 | `4f29b3e3a2b9ea5bffd6748494df2efc748af6b181fd4563bbe31cfa709a4fa8` |
| `moviLLD64.dll` | `3.0.9` | 27,904,776 | `7294c2c00727fef46f2a42c896eb409da2ed3bbf5eab89418ac52dcb749af9ef` |

The compiler reports Movidius Compiler `v00.114.11 Build 4207`, LLVM 14.0.0,
default target `shave`. The backup INF recorded driver version
`10/31/2023,31.0.100.1688`.

## Exports

Confirmed PE AMD64 exports:

| DLL | Tool entry | Cleanup entry |
| --- | --- | --- |
| `moviCompile64.dll` | `main` | `freeResults` |
| `moviAsm64.dll` | `process` | `freeResults` |
| `moviLLD64.dll` | `process` | `freeResults` |

The internal PE export-directory names differ (`moviCompile.dll`,
`moviAsmDll.dll`, and `moviLLDDll.dll`), but the physical `*64.dll` filenames
above are the files loaded.

`freeResults` releases DLL-owned global result state. It is called with no
arguments after every returned buffer has been copied. It is not equivalent
to `free(returned_pointer)`. The DLL is intentionally left loaded until its
private worker process exits, matching the proven lifecycle.

## Compiler effective call

The tested storage/call arrangement is equivalent to:

```c
typedef int (__cdecl *movi_compile_fn)(
    int argc,
    char **argv,
    void *input,
    size_t input_size,
    void **output,
    size_t *output_size,
    void **diagnostic,
    size_t *diagnostic_size);
```

The C source is supplied in the memory buffer; the nominal `entry.c` argument
is only the frontend input name. The first returned pointer/extent pair is the
assembly text. The second pair carried diagnostics in tested failures. All
argv strings, source bytes, and result slots remain alive for the call.

The successful arguments are:

```text
moviCompile.dll
-cc1 -triple shave -target-cpu 3720xx
-S -O2 -x c entry.c
-ffunction-sections -fdata-sections
[-DNAME=DECIMAL ...]
-o -
```

`-cc1` must be the first option after `argv[0]`. `-cc1 -version` is the tested
version query. One null-input `-cc1 --version` trial raised a native access
violation; it does not define a useful API behavior.

The exact declared native width of every compiler extent slot is not proven;
the Windows x64 `size_t` storage above is an execution-tested arrangement.

## Assembler effective call

The tested arrangement is:

```c
typedef int (__cdecl *movi_assemble_fn)(
    int argc,
    char **argv,
    void *input,
    uint32_t input_size,
    void **output,
    size_t *output_size,
    void **diagnostic,
    size_t *diagnostic_size);
```

Static call evidence and the working wrapper agree that the input extent
crosses the exported assembler wrapper as 32 bits. Assembly bytes are supplied
in memory. Working arguments are:

```text
moviAsm.dll --cv 3720xx --noSPrefixing
```

Buffer mode requires both input and output buffers and rejects simultaneous
input/output filenames. `--noFinalSlotCompression` was reported deprecated
and is not part of the working recipe. The first result pair is an ELF32
little-endian SPARC relocatable object; the second pair is treated as
diagnostics.

## Linker effective call

The logical data records are:

```c
struct movi_buffer {
    void *data;
    size_t size;
};

struct movi_buffer_list {
    struct movi_buffer *items;
    size_t count;
};

typedef int (__cdecl *movi_link_fn)(
    int argc,
    char **argv,
    struct movi_buffer_list *inputs,
    struct movi_buffer *output);
```

However, **the leading records alone are not sufficient storage for this DLL**.
The confirmed `npunlock` worker reserves trailing zeroed aggregate storage:

```c
struct {
    struct movi_buffer_list value;
    struct movi_buffer_list trailing[7];
} inputs;

struct {
    struct movi_buffer value;
    struct movi_buffer trailing[7];
} output;
```

The tool receives pointers to the leading `value` records. Supplying only one
pointer/count pair and one pointer/extent pair caused process exception
`0xc0000005`; reserving this trailing storage produced the retained ELF
byte-for-byte. The purpose and true vendor type of the extra storage are not
known. The arrays above are an effective compatibility arrangement, not a
recovered vendor declaration.

Two in-memory inputs are passed in order:

1. the assembler-produced ELF relocatable object;
2. the `shave_kernel.ld` bytes.

`shavecc` supplies the vendored script from embedded library bytes by default,
or uses a non-empty caller-provided view verbatim. The worker protocol still
receives the selected script entirely in memory.

No input filename and no `-T` argument are used. The linker recognizes the
script from the buffer contents. Successful arguments are:

```text
moviLLD.dll -flavor gnu -EL -e <entry-symbol> -z max-page-size=0x10
```

Linker diagnostics were emitted through captured process stdout/stderr rather
than a separate diagnostic result pair.

## Ownership and result handling

For every stage:

1. keep argv, input buffers, aggregate storage, and result slots alive;
2. call the export in a private process;
3. reject nonzero return or invalid/oversized extents;
4. copy returned DLL-owned bytes into caller-owned memory;
5. call that DLL's no-argument `freeResults`;
6. return the copied bytes through the process boundary.

Never free a returned tool pointer with the host CRT and never expose it across
the DLL or process boundary.

## Process and DLL-loading requirements

The entry points behave like command-line programs and may terminate, fault,
or hang their process. Each compile, assemble, and link call therefore runs in
a separate bounded worker with:

- a finite deadline;
- a Windows Job Object with kill-on-close;
- anonymous-pipe IPC carrying versioned length-delimited buffers;
- deterministic termination of descendants;
- no temporary source, assembly, object, script, or ELF files between stages.

The caller supplies an absolute directory. `npunlock` constructs only the
three fixed DLL names beneath it, converts the exact paths to UTF-16, adds only
that directory to the DLL search path, and loads with constrained
`LoadLibraryExW` search flags. Relative Movi DLL names are never resolved
against the current directory or a system-wide guess.

This isolation is a robustness boundary, not a security sandbox. Compiler
arguments should still be constrained because command-style tools may accept
filesystem paths.

## Confirmed data products

For the retained `add1-fp16.c` fixture and linker script:

| Stage | Output |
| --- | --- |
| compile | 2,399-byte assembly, FNV-1a-64 `0b075b77a88d9e73` |
| assemble | 677-byte relocatable ELF, FNV-1a-64 `e704605fae5ddc94` |
| link | 828-byte validated executable ELF |

The complete linked ELF SHA-256 is
`3f9d52273870c2e911000da3c3bd474c0514bd297d29539a14e935b2b01de6d5`.
The native C worker reproduced the retained Python/ctypes result exactly.

The retained linker script is 465 bytes with SHA-256
`4b7faf5233e425c6d75a63b18d7f8ea7e06b3b023cbf3b39731de64294f975da`.
It came from the pinned `npu_compiler` research checkout; MoviTools itself did
not ship a linker script or headers in `MVC_DEPEND`.

## Scope limits

The tested ABI does not establish thread safety, reentrancy, concurrent calls,
inputs larger than the enforced worker bounds, exact unobserved integer widths,
the meaning of linker trailing storage, general option compatibility, other
DLL versions, other targets, or a stable vendor-supported API. Seven runtime
archives exist in `MVC_DEPEND\lib`, but the confirmed MVP kernels do not use
them; linking those archives would require separate unresolved-symbol and
runtime-ABI validation.

## Primary evidence

```text
D:\srcs\level-zero\samples\npurun\NPU_ABI_STATUS.md (sections 16-22)
D:\srcs\level-zero\samples\npurun\experiments\movitools-custom-kernel\
  oem-dll-analysis\README.md
  oem-tool-invocation\README.md
  act-entry-trial\builds\add1-fp16\commands.json
  act-entry-trial\builds\add1-fp16\elf.json
```

The current in-repository implementation and acceptance record are in:

```text
src\workers\movi_worker.c
src\shavecc\shavecc.c
```
