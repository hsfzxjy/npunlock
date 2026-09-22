# npunlock documentation

This wiki explains how to use `npunlock`, how its custom-kernel path works,
and where the current compatibility boundary lies.

## Start here

- [Getting MoviTools](GET_MOVITOOLS.md) — download and extract the external
  compiler toolchain required for custom C kernels.
- [Python API](PYTHON_API.md) — construct, compile, and execute graphs from
  Python.
- [Custom kernels](CUSTOM_KERNELS.md) — write C kernels for the validated
  unary and binary ACT contracts.
- [Current limitations](LIMITATIONS.md) — supported platform, tensor, graph,
  and kernel boundaries.
- [Porting to Linux and newer NPUs](PORTING.md) — unverified compatibility
  hypotheses, suggested experiments, and how to contribute results.

## How it works

- [Intel NPU architecture](INTEL_NPU_ARCHITECTURE.md) — an approachable view
  of the NPU resources relevant to custom kernels.
- [From neural network to machine binary](MODEL_TO_MACHINE_CODE.md) — the
  step-by-step graph serialization, Intel lowering, native binary, and runtime
  execution pipeline, including every major software and hardware component.
- [How npunlock works](HOW_NPUNLOCK_WORKS.md) — carrier graphs, driver graph
  compilation, MoviTools, code substitution, and execution.
- [Development](DEVELOPMENT.md) — native components, building, testing,
  packaging, and public C interfaces.

## Binary-format references

These pages are intended for readers working on the native implementation:

- [Native graph ELF ABI](../ABI/GRAPH_ELF.md)
- [ACT kernel ELF ABI](../ABI/KERNEL_ELF.md)
- [MoviTools DLL contract](../ABI/MOVITOOLS.md)
- [MoviTools math-library symbols](../ABI/MLIBM_SYMBOLS.md)
- [C API guide](../docs/C_API.md)

[Back to the project README](../README.md)
