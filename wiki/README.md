# npunlock documentation

This wiki has two layers: a practical path for writing and running a kernel,
and a reference path for understanding the machinery underneath it. You do
not need to read the binary-format pages before trying the tutorial.

## Pick a path

| If you want to... | Start with... | Then read... |
| --- | --- | --- |
| run your first custom kernel | [Custom-kernel tutorial](TUTORIAL.md) | [Getting MoviTools](GET_MOVITOOLS.md) and [Current limitations](LIMITATIONS.md) |
| look up Python calls | [Python API](PYTHON_API.md) | [Custom kernels](CUSTOM_KERNELS.md) for the C-side contract |
| understand the design | [How npunlock works](HOW_NPUNLOCK_WORKS.md) | [Intel NPU architecture](INTEL_NPU_ARCHITECTURE.md) and [From neural network to machine binary](MODEL_TO_MACHINE_CODE.md) |
| work on the native implementation | [Development](DEVELOPMENT.md) | [C API guide](../docs/C_API.md) and the binary-format references below |
| evaluate a new platform | [Porting roadmap](PORTING.md) | [Porting probe bundle](PORTING_PROBE_BUNDLE.md) |
| check the evidence behind a claim | [Reverse-engineering breakthroughs](REVERSE_ENGINEERING.md) | the relevant ABI reference below |

## Use npunlock

- [Custom-kernel tutorial](TUTORIAL.md) — progress from a first FP16 C loop to
  carrier selection, multi-input kernels, explicit bindings, conversions,
  reusable artifacts, and shared buffers.
- [Getting MoviTools](GET_MOVITOOLS.md) — download and extract the external
  compiler toolchain required for custom C kernels.
- [Python API](PYTHON_API.md) — construct, compile, and execute graphs from
  Python.
- [Custom kernels](CUSTOM_KERNELS.md) — write C kernels for the validated
  unary and binary ACT contracts.
- [Current limitations](LIMITATIONS.md) — supported platform, tensor, graph,
  and kernel boundaries.

## Understand and extend it

- [How npunlock works](HOW_NPUNLOCK_WORKS.md) — carrier graphs, driver graph
  compilation, MoviTools, code substitution, and execution.
- [Intel NPU architecture](INTEL_NPU_ARCHITECTURE.md) — an approachable view
  of the NPU resources relevant to custom kernels.
- [From neural network to machine binary](MODEL_TO_MACHINE_CODE.md) — the
  step-by-step graph serialization, Intel lowering, native binary, and runtime
  execution pipeline, including every major software and hardware component.
- [Reverse-engineering breakthroughs](REVERSE_ENGINEERING.md) — the controlled
  experiments that established custom code compilation, ACT invocation, graph
  substitution, multi-input kernels, and precision-preserved FP32 execution.
- [Development](DEVELOPMENT.md) — native components, building, testing,
  packaging, and public C interfaces.
- [Porting to Linux and newer NPUs](PORTING.md) — unverified compatibility
  hypotheses, suggested experiments, and how to contribute results.
- [Porting probe bundle](PORTING_PROBE_BUNDLE.md) — package one patched graph,
  deterministic tensors, provenance, hashes, and an exact host oracle.

## Binary-format references

These pages preserve the exact observed contracts, evidence boundaries, and
implementation details. They are for readers working on the native code or
checking a low-level claim; most users can stop at the pages above.

- [Native graph ELF ABI](../ABI/GRAPH_ELF.md)
- [ACT kernel ELF ABI](../ABI/KERNEL_ELF.md)
- [MoviTools DLL contract](../ABI/MOVITOOLS.md)
- [MoviTools math-library symbols](../ABI/MLIBM_SYMBOLS.md)
- [C API guide](../docs/C_API.md)

[Back to the project README](../README.md)
