# Porting to Linux and newer NPUs

[Documentation index](README.md)

`npunlock` is currently verified only on Windows x64 with Meteor Lake /
NPU3720. This page describes promising routes beyond that baseline. They are
porting hypotheses and test plans, not supported configurations.

It helps to separate three compatibility questions:

1. Can the NPU execute the compiled SHAVE instructions and use the same kernel
   invocation ABI?
2. Will the driver and firmware accept the patched native graph blob?
3. Can the host run the graph compiler, MoviTools, and execution workers?

A positive answer to one does not establish the others.

## Porting utilities and staged roadmap

Porting is intentionally split into independently testable milestones. A
failure in one stage must not be reported as evidence about another stage.

### Milestone 1: offline native-blob inspection

`npunlock-inspect` builds on Linux and Windows without MoviTools, Level Zero,
an Intel NPU, or the worker executable. It hashes a caller-provided native blob
and applies the same bounds-checked ACT discovery used by `patchblob`:

```bash
cmake --preset linux-inspect
cmake --build --preset linux-inspect
./build/linux-inspect/npunlock-inspect \
  --graph patched.blob \
  --report inspect.json
```

The JSON report records the blob size and SHA-256 plus every discovered ACT
group, invocation/range index, arity, element count, byte span, precision, and
contract flags. Discovery means only that the file matches the narrow observed
NPU3720/compiler-8.3 structures. It does not prove that a Linux driver, newer
firmware, or different NPU can load or execute the blob.

### Milestone 2: reproducible probe bundle

The standard-library-only `tools/porting_probe.py` helper creates and verifies
a small redistributable directory containing a patched native blob, raw input,
exact expected output, tensor selectors, hashes, and required Windows
driver/compiler provenance. The first schema is deliberately fixed to static
dense `1x16` FP16 add-one. See the
[probe-bundle contract](PORTING_PROBE_BUNDLE.md). The bundle contains no
MoviTools or driver binaries.

### Milestone 3: minimal Linux Level Zero execution

`npunlock-linux-probe` now reuses the graph-inference execution engine,
dynamically loads the system Level Zero loader, and consumes a verified probe
bundle containing an existing native blob. It reports loader discovery, NPU
selection, graph-extension discovery, graph creation, initialization,
execution, and host-oracle comparison separately. Run it after verifying the
bundle:

```bash
python3 tools/porting_probe.py verify build/add1-probe
./build/linux-inspect/npunlock-linux-probe \
  --bundle build/add1-probe \
  --report build/add1-linux-report.json
```

This first bring-up path runs in-process and uses a finite fence wait. A driver
call that never returns cannot be terminated independently. WSL testing has
confirmed loader discovery and a clean `npu-not-found` result at device
selection, but WSL has no NPU passthrough. Graph creation, execution, and the
oracle therefore still require a real Linux NPU system before this milestone
can be considered hardware-validated.

### Milestone 4: POSIX worker isolation

A reusable bounded POSIX process launcher is now implemented and tested
offline. It uses a dedicated binary-response descriptor, nonblocking request
transport, separate stdout and stderr pipes, a child process group, finite
deadlines, and deterministic group termination. The focused fixture verifies
both byte-exact transport and timeout cleanup.

The next slice is to build the unified `npunlock_worker infer` child on Linux
and connect copied graph inference to this launcher. Until then,
`npunlock-linux-probe` remains the explicitly in-process bring-up path. Shared
Level Zero buffers may remain the same narrow in-process exception as Windows.

### Milestone 5: Linux graph compilation

Port the Level Zero `NGRAPH_LITE` graph-compilation path and record the compiler
extension/version, device identity, build flags, and native-blob hash. Compare
Linux- and Windows-produced blobs structurally, but do not require them to be
byte-identical.

### Deferred: MoviTools invocation on Linux

MoviTools remains a Windows PE/DLL toolchain and is not part of the native
Linux port. Initially compile SHAVE ELF files on Windows and transfer them. A
future Wine-hosted worker is a separate experiment and must not be presented as
native Linux support until validated.

## Linux with NPU3720

The working hypothesis is that a patched NPU3720 graph blob generated on
Windows can also run on Linux. The custom machine code is executed by the NPU,
and the final graph binary is consumed by the device firmware rather than by
the Windows CPU. The custom kernel itself therefore has no Windows API or host
CRT dependency.

This has not been tested. Native graph blobs may still be sensitive to the
driver, graph-compiler, or firmware version, and a Linux driver may reject a
blob produced by a different software stack. Successful loading is also not a
correctness result; output must be checked against a host oracle.

The current host tooling is Windows-only. There are two separable Linux tasks:

- **Run an existing patched blob.** Add a Linux Level Zero graph loader and
  execution worker, then test a blob produced and patched on Windows.
- **Build the kernel on Linux.** The available MoviTools components are Windows
  DLLs. A Linux port would need a reliable way to load those PE DLLs, such as a
  suitable compatibility environment, or compile the SHAVE ELF on a Windows
  machine and transfer that artifact. Neither route has been validated here.

An initial Linux/NPU3720 experiment should keep every other variable fixed:

1. Generate and validate a simple patched blob on the current Windows path.
2. Load that exact blob through the Linux Intel NPU Level Zero graph extension.
3. Run fixed inputs and compare every output with the same host oracle.
4. Record the blob hash plus Linux driver, firmware, and device versions.

## Newer Intel NPUs

Two routes are worth investigating on NPU4000 and later generations.

### Route 1: try the existing NPU3720 SHAVE image

It is possible that machine code produced for the MoviTools `3720xx` target
remains executable on a newer NPU. This is plausible enough to test, but no
instruction-set or ABI compatibility is currently claimed.

Even if the instruction stream executes, a newer generation may use different
ACT invocation records, tensor layouts, memory rules, carrier lowering, or
native graph structures. The existing `npunlock/npu3720_kernel.h` assumptions
and graph patch locations must not be reused without validation. Start with a
small unary kernel and a host oracle before testing nonlinear or multi-input
kernels.

### Route 2: find generation-matched MoviTools

An older OEM driver package that advertises support for NPU4000 or a later
generation may still contain the MoviTools DLLs, target definitions, and
libraries needed to produce a matching SHAVE ELF. A useful investigation would:

1. Extract the package without installing or downgrading the active driver.
2. Identify its MoviTools versions, hashes, supported target selectors, and
   math libraries.
3. Compile a minimal kernel and validate the resulting ELF rather than assuming
   that a new target name implies compatibility.
4. Characterize the new invocation and graph contracts before adding a
   generation-specific kernel header or patch path.

MoviTools remains proprietary. Do not commit, redistribute, or attach extracted
DLLs and libraries to project issues.

## How to contribute results

Contributions from people with Linux systems, NPU4000+ hardware, or other Intel
NPU configurations are welcome. Useful reports include:

- OS, CPU/NPU model, PCI device ID, and driver/firmware versions;
- the exact build and execution commands;
- graph blob and SHAVE ELF hashes;
- whether graph creation, execution, and output validation each succeeded;
- output comparison against a host oracle; and
- complete diagnostics, stdout, and stderr, with machine-local paths or private
  data removed.

Please report failures too. A rejected blob, changed ACT layout, or incompatible
ELF is valuable evidence and helps turn these hypotheses into a real
compatibility matrix.

For the currently verified boundary, see [Current limitations](LIMITATIONS.md).

[Back to documentation index](README.md)
