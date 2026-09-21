# npunlock

`npunlock` is an experimental toolchain for running **user-written C kernels on Intel Core Ultra NPUs**, without depending on the OpenVINO runtime.

Intel NPUs are normally programmed through graph-level software such as OpenVINO. An application provides a neural-network model, Intel's compiler maps that model onto the NPU, and the driver executes the resulting native graph.

What this software stack does **not** expose is a public low-level workflow for writing a C kernel and executing it directly on the NPU's programmable **ACT-SHAVE** processors. Developers can describe operations as part of a model graph, but they are not normally given an interface equivalent to:

```text
kernel.c
   ↓
compile
   ↓
run on ACT-SHAVE
```

`npunlock` provides that missing path.

On Meteor Lake / NPU3720, it can compile user-provided C into a SHAVE executable using the older Movidius-derived MoviTools toolchain, integrate that executable into an NPU graph, and execute the resulting graph on the hardware.

`npunlock` also does not require OpenVINO to construct the graph. Intel's NPU user-mode driver can accept OpenVINO-format IR directly and compile it into the native graph representation consumed by the NPU runtime.

Together, these two paths provide an end-to-end workflow:

```text
                 graph IR
                    │
                    ▼
          Intel NPU user-mode driver
                    │
                    ▼
             native graph blob
                    │
                    │
user C              │
   │                │
   ▼                │
MoviTools           │
   │                │
   ▼                │
SHAVE executable ───┘
         │
         ▼
replace selected ACT kernel code
         │
         ▼
patched native graph
         │
         ▼
      Intel NPU
```

The result is a complete custom-kernel path:

```text
C source
  → SHAVE machine code
  → native NPU graph
  → NPU execution
```

OpenVINO IR is used as an input format for graph compilation, but the OpenVINO runtime itself is not part of this pipeline.

The research prototype behind `npunlock` has already demonstrated this path on NPU3720 with custom nonlinear computation and an observed multi-input ACT-SHAVE kernel layout. The standalone project turns those findings into a small, reproducible tool with explicit validation and well-defined failure behavior.

## Intel NPU architecture

An NPU, or **Neural Processing Unit**, is a processor specialized for neural-network workloads. Intel Core Ultra processors integrate an NPU alongside the CPU and GPU to provide dedicated, power-efficient local AI execution.

The Intel NPU is not a single homogeneous processor. At a high level, two types of compute resources are particularly relevant to `npunlock`:

```text
                    Intel NPU

        ┌─────────────────────────────┐
        │   DPU / tensor engines      │
        │                             │
        │   convolution               │
        │   matrix multiplication     │
        │   regular tensor compute    │
        └──────────────┬──────────────┘
                       │
                       │ tensors
                       │
        ┌──────────────▼──────────────┐
        │   ACT-SHAVE processors      │
        │                             │
        │   programmable code         │
        │   software operators        │
        │   irregular computation     │
        └─────────────────────────────┘
```

The **DPU** provides specialized hardware for the dense, regular tensor operations that account for much of a neural network's computation.

The **ACT-SHAVE** processors provide a programmable execution path for software-defined operations that are not naturally handled by the DPU.

Applications normally do not need to distinguish between these resources. The NPU compiler decides how each part of a model should be mapped onto the hardware.

`npunlock` exposes the programmable ACT-SHAVE path directly.

## How software normally reaches the NPU

A typical application operates entirely at the model level:

```text
model
  │
  ▼
OpenVINO
  │
  ▼
Intel NPU compiler
  │
  ▼
native graph
  │
  ▼
driver + firmware
  │
  ▼
NPU
```

The NPU compiler does substantially more than translate individual mathematical operations. It constructs an execution graph describing compute tasks, tensor placement, synchronization, data movement, and the software kernels needed by the workload.

This abstraction is appropriate for normal inference applications. Developers do not need to know how individual ACT-SHAVE kernels are represented, how NPU tasks are scheduled, or how the native graph is encoded.

The tradeoff is that ACT-SHAVE programmability is not exposed as a conventional user-facing kernel API.

## The custom-kernel gap

Intel's NPU software stack already relies on ACT-SHAVE code internally. Native graphs can contain executable software kernels, and those kernels are scheduled alongside DPU tasks as part of the same graph.

What is missing from the public programming model is the producer side of that interface:

```text
my_kernel.c
     │
     ▼
ACT-SHAVE compiler
     │
     ▼
custom kernel binary
     │
     ▼
NPU graph
```

Older Intel/Movidius software packages contain a toolchain known as **MoviTools**, including components such as:

```text
moviCompile64.dll
moviAsm64.dll
moviLLD64.dll
```

These tools descend from the Movidius SHAVE development toolchain.

The key result behind `npunlock` is that this older compiler can still produce SHAVE executables that are usable by the Meteor Lake NPU3720 software stack.

This supplies the missing compiler path from user C to executable ACT-SHAVE code.

## How npunlock works

Intel does not provide a public toolchain for taking user-written C and directly turning it into a custom ACT-SHAVE kernel inside an NPU graph. `npunlock` therefore takes an indirect but practical route: it lets Intel's compiler create a valid ACT execution environment around a known kernel first, then replaces that kernel's executable code with the user-compiled implementation.

This avoids having to reconstruct Intel's native graph format or reproduce the work performed by the NPU compiler.

The process has four stages.

### 1. Build a graph with a known ACT kernel

`npunlock` starts from an OpenVINO-format IR in which the desired custom operation is temporarily represented by a **known operation that Intel's NPU compiler already lowers to ACT-SHAVE**.

That known operation acts as a carrier.

Conceptually:

```text
desired graph

input
  │
  ▼
custom operation
  │
  ▼
output
```

is initially represented as:

```text
carrier graph

input
  │
  ▼
known ACT-SHAVE operation
  │
  ▼
output
```

At this point there is no custom machine code in the graph. The purpose of the carrier is simply to give Intel's compiler an operation it already knows how to turn into a valid ACT-SHAVE task.

### 2. Let Intel's driver compile the carrier graph

The carrier IR and its weights are submitted directly to the Intel NPU user-mode driver:

```text
carrier IR + weights
        │
        ▼
Intel NPU user-mode driver
        │
        ▼
native graph blob
```

Intel's compiler performs the hardware-specific work: it constructs the graph tasks, memory layout, synchronization, relocations, and the ACT-SHAVE invocation for the carrier operation.

`npunlock` then retrieves the compiled native graph blob from the driver.

This is an important part of the design. `npunlock` does not attempt to reproduce Intel's graph compiler. Instead, it asks Intel's own compiler to construct a valid execution environment first and preserves that environment when inserting the custom kernel.

### 3. Compile the custom kernel separately

The user's C source is compiled independently using MoviTools:

```text
custom kernel.c
       │
       ▼
   moviCompile
       │
       ▼
     assembly
       │
       ▼
     moviAsm
       │
       ▼
      object
       │
       ▼
     moviLLD
       │
       ▼
linked ACT-SHAVE ELF
```

The resulting ELF contains the executable SHAVE code for the custom operation.

Intel's graph compiler never sees this C source and does not need to understand the custom operation.

### 4. Replace the carrier kernel

`npunlock` locates the compatible ACT-SHAVE invocation group at the custom
operation's validated position in the compiled native graph. Explicit
invocation/range selection remains available as an advanced override.

It extracts the executable image from the custom SHAVE ELF, adds it to the graph blob, and updates the selected ACT kernel metadata and code relocations so that the existing invocation executes the new code.

Conceptually:

```text
        graph produced by Intel

        ┌──────────────────────┐
        │ DPU / DMA tasks      │
        │ memory layout        │
        │ barriers             │
        │                      │
        │ ACT invocation       │
        │       │              │
        │       ▼              │
        │ carrier kernel code  │
        └──────────────────────┘
                    │
                    │ replace code target
                    ▼
        ┌──────────────────────┐
        │ DPU / DMA tasks      │
        │ memory layout        │
        │ barriers             │
        │                      │
        │ ACT invocation       │
        │       │              │
        │       ▼              │
        │ custom SHAVE code    │
        └──────────────────────┘
```

The surrounding graph remains the graph produced by Intel's compiler. `npunlock` changes the executable code associated with the selected ACT task rather than rebuilding the graph itself.

The complete path is therefore:

```text
                 carrier graph
                      │
                      ▼
               Intel NPU UMD
                      │
                      ▼
              native graph blob
                      │
                      │
user C                │
   │                  │
   ▼                  │
MoviTools             │
   │                  │
   ▼                  │
custom SHAVE ELF      │
   │                  │
   └─────────┬────────┘
             │
             ▼
      replace carrier kernel
             │
             ▼
      patched native graph
             │
             ▼
          Intel NPU
```

This division is central to `npunlock`.

Intel's compiler remains responsible for constructing the native graph, scheduling tasks, arranging memory, and defining the ACT invocation environment. MoviTools provides the missing path from user C to ACT-SHAVE machine code. `npunlock` connects the two by replacing the code of a compatible compiler-generated ACT kernel while preserving its surrounding execution contract.

The positional mapping is intentionally narrow. Automatic Python selection is
enabled only when the native ACT group count matches the complete topological
computational-node count and every selected custom group has a compatible
observed tensor contract. DPU-containing, fused, optimized-away, or otherwise
count/arity-mismatched graphs fail closed and require explicit patch targets.
This does not claim a general mapping from arbitrary source node names to
native ACT invocations.

## What custom kernels enable

Most neural-network computation should remain on the NPU's dedicated tensor hardware. DPU execution is the appropriate path for operations such as convolution and matrix multiplication.

Custom ACT-SHAVE kernels address a different class of work: computation that benefits from programmability rather than fixed-function tensor throughput.

Potential uses include:

- implementing operators not supported by the existing NPU compiler;
- keeping model-specific computation inside the NPU graph instead of falling back to the CPU;
- combining small operations into a custom unified kernel;
- implementing specialized numerical routines;
- investigating ACT-SHAVE execution and the broader NPU architecture.

The goal is therefore not to replace the DPU with programmable code. It is to make the programmable component that already exists within the NPU accessible to user software.

## Current results

The research prototype has demonstrated the complete basic path on Meteor Lake / NPU3720:

```text
user C
   ↓
SHAVE executable
   ↓
native graph integration
   ↓
NPU execution
   ↓
verified custom computation
```

The demonstrated cases include nontrivial nonlinear computation and an observed multi-input ACT-SHAVE layout, in addition to simpler unary kernels.

The standalone `npunlock` project is focused on turning this proof of concept into a reproducible implementation with explicit interfaces, validation, diagnostics, and bounded failure behavior.

The initial scope is intentionally narrow:

```text
Host             Windows x64
NPU              Meteor Lake / NPU3720
SHAVE target     3720xx
Tensor scope     static dense FP16; validated unary FP32 ACT in accuracy mode
Kernel image     one compiled SHAVE image
Patch selection  validated positional ACT groups; explicit range override
```

These constraints describe the contracts that have been established so far. They are not intended to imply limitations of the underlying hardware.

Dynamic shapes, arbitrary tensor layouts and other data types, arbitrary node-name
mapping, and additional NPU generations remain outside the initial MVP.

## Project scope

`npunlock` is not a replacement for OpenVINO.

It is not a general compiler for the NPU's DPU hardware.

It is not currently a CUDA-like programming model for the complete NPU.

And it does not assume that arbitrary C programs can be executed safely or efficiently as NPU kernels.

Its initial objective is narrower:

> provide a reproducible path from user-written C to executable ACT-SHAVE code inside a valid Intel NPU graph, while relying on Intel's existing driver and firmware for graph compilation and execution.

The project deliberately exposes only the contracts that have been observed and validated. Unsupported graph structures, kernel layouts, or ABI assumptions should be rejected rather than handled speculatively.

## Build and offline verification

The supported host build is Windows x64 with MSVC. The public CMake preset does
not select a particular Visual Studio release or generator. Run every
MSVC-dependent command through the repository wrapper so the latest installed
C++ build environment is normalized consistently:

```powershell
python -m pip install .
tools\msvc-run.ps1 cmake --preset windows
tools\msvc-run.ps1 cmake --build --preset windows-debug
tools\msvc-run.ps1 ctest --preset windows-debug
tools\msvc-run.ps1 cmake --build build\windows --config Debug --target format-check
```

The default configuration is offline and does not invoke MoviTools, OpenVINO,
the Intel NPU driver, or NPU hardware. It does run the byte-exact patch proof
against the small licensed fixtures under `tests/fixtures/npu3720`. Python
installation is needed only for the Python frontend test; CMake skips that
test when Python or NumPy is absent.

The MoviTools and hardware suites remain separately gated. Their source, IR,
graph, and golden-output fixtures are bundled. CMake never stores the local
proprietary DLL path; the enabled MoviTools test reads it from the runtime
environment:

```powershell
$env:NPUNLOCK_MOVITOOLS_DIR = 'D:\path\containing\MoviTools\DLLs'
tools\msvc-run.ps1 cmake --preset windows -B build\integration `
  -DNPUNLOCK_ENABLE_MOVITOOLS_TESTS=ON `
  -DNPUNLOCK_ENABLE_NPU_TESTS=ON `
  -DNPUNLOCK_ENABLE_GRAPHINFER_TESTS=ON
tools\msvc-run.ps1 cmake --build build\integration --config Debug
tools\msvc-run.ps1 ctest --test-dir build\integration -C Debug --output-on-failure
```

Install the component targets, single native DLL, public headers, CLI, unified
worker, and CMake package with:

```powershell
tools\msvc-run.ps1 cmake --install build\windows --config Debug --prefix build\stage
```

See [`docs/C_API.md`](docs/C_API.md) for C ownership, linking, deployment, and
per-library usage contracts. The experimentally observed binary contracts are
documented separately in [`ABI/GRAPH_ELF.md`](ABI/GRAPH_ELF.md),
[`ABI/KERNEL_ELF.md`](ABI/KERNEL_ELF.md), and
[`ABI/MOVITOOLS.md`](ABI/MOVITOOLS.md); those references distinguish confirmed
behavior from inference and open questions.

## Current MVP command

`npurun build` is the file-oriented boundary over the four C libraries. It
compiles an OpenVINO-format IR through the NPU driver, compiles the supplied C
through caller-selected MoviTools DLLs, discovers and validates the selected
positional ACT group, then writes the graph and provenance manifest:

```powershell
npurun build `
  --ir model.xml `
  --weights model.bin `
  --shave-source kernel.c `
  --movi-dll-dir D:\path\containing\MoviTools\DLLs `
  --patch-position 0 `
  --output patched.blob `
  --manifest patched.json
```

`--patch-position` is a zero-based ACT operation-group position, not a source
node name. Discovery derives all compatible invocations and their observed
arity/span contract. The explicit `--patch-invocation`, `--patch-range`,
`--input-count`, `--element-count`, and `--span-bytes` options remain available
for advanced overrides. `shavecc` uses its vendored NPU3720 linker script by
default; `--linker-script FILE` supplies an exact caller override.
Hardware/OEM calls have finite worker deadlines.
MoviTools binaries remain caller-supplied and are not bundled with this project.
When `--movi-dll-dir` is omitted, `npurun` reads
`NPUNLOCK_MOVITOOLS_DIR`; the CLI argument takes precedence when both are set.

For the narrow validated unary add-one demonstration, `npurun` can also load
and execute the patched graph in a bounded private process, save the raw FP16
output, and require exact agreement with its host oracle before writing any
artifacts:

```powershell
npurun build `
  # the same build and explicit patch options shown above `
  --run-add1 `
  --run-input input.fp16 `
  --run-output output.fp16
```

The CLI oracle remains intentionally operator-specific. General caller-provided
native graph and tensor execution is available through the public `graphinfer`
C library. `graphinfer` selects inputs by explicit graph-argument index or exact
name, requires the caller to cover every graph input, and returns all output
metadata and buffers through library-owned allocations. Its current graph-I/O
contract is bounded static dense FP16 or FP32 tensors; driver calls execute in
a finite-lived worker.

## Python frontend

The `python/npunlock` package provides the initial symbolic frontend. Operators
are dynamic rather than registered:

```python
import npunlock as npu

x = npu.input("x", shape=(1, 32), dtype="f16")
y = npu.Abs(x, _shape=x.shape, _dtype=x.dtype)
graph = npu.Graph([x], [y])
serialized = npu.serialize_ir(graph)
xml, weights = serialized.xml, serialized.weights
```

The package serializes OpenVINO-format IR without importing OpenVINO and binds
the four public C APIs from the bundled `npunlock.dll` directly through
`ctypes`. Its wheel also bundles `npunlock_worker.exe`, and that package-local
directory is the default `native_dir`. An explicit `native_dir=` or
`NPUNLOCK_NATIVE_DIR` remains available for development overrides. Custom
nodes lower to caller-selected carrier operators. Automatic patch selection is
used only when the complete topological computational-node sequence maps
one-to-one to the validated positional ACT groups and the selected custom
groups have matching arity. Explicit advanced `PatchTarget` values remain the
escape hatch for other validated graph layouts.
Configure MoviTools for custom compilation either in Python or through the
runtime environment:

```python
npu.configure(movi_dll_dir=r"D:\path\containing\MoviTools\DLLs")

# Alternatively, set NPUNLOCK_MOVITOOLS_DIR before calling npu.compile().
```

An explicit `movi_dll_dir=` passed to `npu.compile()` overrides both settings;
the process-local `configure()` value overrides the environment.
The directory contains the three MoviTools DLLs. Math functions are resolved
from the same distribution's sibling `..\lib\mlibm.a`; `npunlock` does not
redistribute that proprietary archive.
`Program.run()` validates named NumPy FP16/FP32 inputs against the symbolic graph,
calls `graphinfer`, and returns copied NumPy outputs without implementing a
second Level Zero execution path in Python.

FP32 custom nodes are marked with OpenVINO's serialized
`DisablePrecisionConversion` runtime attribute. `npu.compile()` also selects
the compiler's `EXECUTION_MODE_HINT="ACCURACY"` when no build flags were
provided. Together these retain a single FP32 ACT group instead of silently
lowering the carrier through FP32-to-FP16-to-FP32 conversion. Caller-provided
build flags for an FP32 custom graph must explicitly contain the same accuracy
hint.

[`examples/example_multilayer_multi_input.py`](examples/example_multilayer_multi_input.py)
demonstrates two independently bound FP16 inputs, separate unary preprocessing
branches, a two-input custom kernel, and an ordinary postprocessing layer. The
exact `(1, 32)` graph is intentionally retained because its compiler-8.3 ACT
group sequence and binary parameter layout have been validated; the example
does not imply broadcasting or arbitrary binary layouts.
