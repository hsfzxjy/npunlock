# MoviTools DLL invocation contract

Status: experimental. Last updated 2026-09-21.

This document explains how `shavecc` calls the Windows MoviTools compiler,
assembler, and linker DLLs. These DLL exports behave like in-memory versions
of command-line programs; they are not documented as a stable public SDK ABI.

The declarations below are compatibility definitions that work with the
tested tool versions. They should not be copied into a general-purpose wrapper
without the same process isolation, ownership rules, and output validation.

## What MoviTools does in `npunlock`

The three tools form a conventional compiler pipeline:

```text
C source
  -> moviCompile64.dll -> SHAVE assembly
  -> moviAsm64.dll     -> relocatable ELF32 object
  -> moviLLD64.dll     -> linked ELF32 executable
```

`npunlock` exchanges every stage as a memory buffer. It does not create hidden
source, assembly, object, linker-script, or ELF temporary files.

The tested versions are:

| DLL | Reported version |
| --- | --- |
| `moviCompile64.dll` | `00.114.11.4207` |
| `moviAsm64.dll` | `1.13.16 64-bit` |
| `moviLLD64.dll` | `3.0.9` |

The compiler identifies itself as Movidius Compiler v00.114.11 Build 4207,
based on LLVM 14.0.0. Other releases may have different undocumented calling
details and must be validated separately.

## Supplying the tool directory

MoviTools is proprietary and is not distributed with `npunlock`. The caller
must supply the `MVC_DEPEND` root. Its `bin` directory contains:

```text
moviCompile64.dll
moviAsm64.dll
moviLLD64.dll
```

The linker also uses `mlibm.a` from the root's `lib` directory when a
kernel references supported math functions.

With the tested compiler and archive, most conventional `libm` functions can
be referenced by their normal C names without including `<math.h>` in the
kernel source. This is observed toolchain behavior, not a claim that every host
`libm` symbol is present. A function is usable only when MoviTools accepts the
call, `mlibm.a` resolves it, and the linked ELF remains within the validated
kernel contract.

The separate [`mlibm.a` symbol inventory](MLIBM_SYMBOLS.md) contains all
externally defined names observed in the tested archive. These are symbols
that may be referenced, not confirmed prototypes. The archive does not encode
C signatures; likely signatures can be inferred from conventional function
names or existing `libm` implementations and then verified experimentally.

The public interfaces accept this directory through:

- the C `shavecc_options` structure;
- `npurun --movi-dll-dir` or `NPUNLOCK_MOVITOOLS_DIR`; or
- Python `npu.configure(...)`, `npu.compile(movi_dll_dir=...)`, or the same
  environment variable.

There is no built-in machine path. The implementation constructs only the
three fixed `bin` paths and the required `lib\mlibm.a` path beneath the
supplied root. Passing `bin` itself is not supported. It does not search the
current directory, driver installation, or arbitrary system locations.

## Exported entry points

The tested PE/AMD64 exports are:

| DLL | Main entry | Result cleanup |
| --- | --- | --- |
| `moviCompile64.dll` | `main` | `freeResults` |
| `moviAsm64.dll` | `process` | `freeResults` |
| `moviLLD64.dll` | `process` | `freeResults` |

The physical files use the `*64.dll` names above. Their internal export-module
names omit or vary the `64` suffix; callers should load the physical filenames
and resolve the named exports directly.

`freeResults` takes no arguments and releases result state owned by the DLL.
It is not equivalent to calling the host C runtime's `free()` on an output
pointer.

## Compiler call

The execution-tested compiler signature is equivalent to:

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

The input buffer contains C source. `entry.c` is only the name presented to
the frontend for diagnostics. The first returned pointer/size pair contains
assembly text; the second contains diagnostic text in observed failures.

The working arguments are:

```text
moviCompile.dll
-cc1 -triple shave -target-cpu 3720xx
-S -O2 -x c entry.c
-ffunction-sections -fdata-sections
[-DNAME=DECIMAL ...]
-o -
```

`-cc1` must be the first option after `argv[0]`. The accepted public definition
syntax is intentionally limited to uppercase names and decimal values. All
argument strings, source bytes, and result slots remain alive until the call
returns.

The exact vendor declaration for every size field is unavailable. Windows x64
`size_t` storage is the tested arrangement.

## Assembler call

The execution-tested assembler signature is equivalent to:

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

Unlike the compiler wrapper, the assembler input length is passed as a 32-bit
value. The input buffer contains assembly text. The first result is an ELF32,
little-endian relocatable object; the second result is diagnostic text.

Working arguments are:

```text
moviAsm.dll --cv 3720xx --noSPrefixing
```

Buffer mode requires memory input and output. Input or output filenames must
not be supplied at the same time.

## Linker call

The visible leading records look like this:

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

The DLL accesses more aggregate storage than these leading fields describe.
Supplying only the visible records caused an access violation. The tested
compatibility layout reserves seven additional zeroed records after each
leading value:

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

The linker receives pointers to `inputs.value` and `output.value`. The purpose
and official vendor type of the trailing storage are unknown. This is a tested
compatibility arrangement, not a recovered vendor header.

Two input buffers are passed in order:

1. the assembler-produced relocatable ELF object;
2. the selected linker-script bytes.

No input filename and no `-T` option are used. The DLL recognizes the script
from the second buffer. Working arguments are:

```text
moviLLD.dll -flavor gnu -EL -e <entry-symbol> -z max-page-size=0x10
  --gc-sections <MoviTools-root>\lib\mlibm.a
```

`shavecc` embeds
[`src/shavecc/shave_kernel.ld`](../src/shavecc/shave_kernel.ld) as its default
script. A nonempty caller-supplied script replaces it verbatim.

`--gc-sections` is required for the supported math-library path. Without it,
an otherwise simple math function may retain unrelated archive members and a
writable global in `.arg.data`. Such a kernel cannot be installed by the
current graph patcher. Garbage collection keeps only the reachable code and
allows validated math kernels to remain self-contained in `.text`.

Linker diagnostic messages are captured from the worker process's standard
output and standard error streams.

## Ownership rules

Each stage follows the same lifecycle:

1. allocate and keep alive all arguments, input buffers, aggregate storage,
   and result slots;
2. invoke the DLL entry point inside an isolated worker process;
3. reject a nonzero return code or an invalid/oversized result;
4. copy DLL-owned result bytes into `npunlock`-owned memory;
5. call that DLL's no-argument `freeResults`; and
6. return only the copied bytes across the process boundary.

Never pass a DLL result pointer to the host CRT, and never expose that pointer
through the public C ABI. The process exits after cleanup, which also discards
any undocumented global state held by the tool.

## Why every stage uses an isolated worker

The exports behave like command-line program entry points. Invalid input or an
internal failure may hang, fault, or terminate the calling process. Running
them inside the application or Python process would make those failures
unrecoverable.

Each stage therefore runs with:

- a finite deadline;
- a Windows Job Object configured to terminate descendants on close;
- anonymous-pipe IPC with versioned, length-delimited messages;
- bounded input and output sizes; and
- deterministic process cleanup.

This provides failure containment, not a security sandbox. Tool arguments are
still constrained, and callers should use only trusted MoviTools installations.

## DLL loading

The selected DLL path is converted to an absolute UTF-16 path. The worker adds
only its `bin` directory to the DLL search path and uses constrained
`LoadLibraryExW` flags. Relative dependency names are resolved within that
controlled search configuration rather than against the working directory.

The tool files and `mlibm.a` remain external dependencies. `npunlock` never
modifies or redistributes them.

## Unsupported assumptions

The tested calls do not establish:

- a stable vendor-supported ABI;
- compatibility with arbitrary MoviTools versions;
- thread safety or reentrancy inside one process;
- the official meaning of the linker's trailing storage;
- exact unobserved integer widths;
- inputs larger than `npunlock`'s worker limits;
- general compatibility of additional compiler/linker options;
- safe use of other runtime archives; or
- targets other than `3720xx`.

Any tool-version change should be treated as a new compatibility target and
revalidated through the complete compile, ELF-validation, patch, and hardware
execution path.

## Related public documentation

- [KERNEL_ELF.md](KERNEL_ELF.md) describes the ELF that the linker must return.
- [MLIBM_SYMBOLS.md](MLIBM_SYMBOLS.md) inventories the tested math archive.
- [GRAPH_ELF.md](GRAPH_ELF.md) describes how its `.text` image is installed in
  a native graph.
- [C API](../docs/C_API.md) documents the public `shavecc` interface.
- [`movi_worker.c`](../src/workers/movi_worker.c) contains the compatibility
  declarations and isolated stage implementation.
