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
