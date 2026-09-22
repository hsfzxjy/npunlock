# npunlock

**Custom C kernels for Intel Core Ultra NPUs.**

`npunlock` lets you write custom operations in C and run them inside a graph on
an Intel Core Ultra NPU. Intel normally exposes the NPU through model- and
graph-level APIs; `npunlock` opens a path to the programmable processors behind
some of those graph operations.

The currently verified platform is Windows x64 with Meteor Lake / NPU3720. The
installed Intel NPU driver still compiles the surrounding graph and handles
hardware execution—`npunlock` adds the custom code path.

```text
Python graph + C kernel
          |
          v
       npunlock
          |
          v
 Intel NPU driver
          |
          v
         NPU
```

## Quick example

This is the computation at the heart of the runnable GELU kernel:

```c
/* Abbreviated computation body: not a complete ACT entry point. */
for (unsigned i = 0; i < count; ++i) {
    float x = (float)in[i];
    float w = x + 0.044715f * x * x * x;
    w = tanhf(w * 0.79788456f);
    out[i] = (__fp16)(0.5f * x * (1.0f + w));
}
```

The complete source wraps this loop in the required entry point and tensor
descriptor handling. Python places that C kernel inside an NPU graph:

```python
import numpy as np
import npunlock as npu

from examples.example_gelu import gelu_c

npu.configure(movi_dll_dir=r"C:\path\to\MVC_DEPEND")

x = npu.input("x", shape=(1, 2048), dtype="f16")
y = npu.custom(
    x,
    source=gelu_c,
    carrier="Abs",
    _shape=x.shape,
    _dtype=x.dtype,
    _name="gelu",
)

program = npu.compile(npu.Graph(inputs=[x], outputs=[y], name="gelu_example"))
input_value = np.linspace(-4, 4, 2048, dtype=np.float16).reshape(1, -1)
output = program.run({"x": input_value})["gelu"]
```

In short: the C computation becomes NPU machine code and runs as part of the
graph. See the complete, runnable [FP16 GELU example](examples/example_gelu.py),
plus the [FP32 GELU](examples/example_gelu_f32.py) and
[multi-layer two-input](examples/example_multilayer_multi_input.py) examples.

## Why npunlock?

OpenVINO and Intel's public graph APIs let applications submit operations that
the graph compiler understands. They do not provide a normal user-facing path
equivalent to:

```text
kernel.c -> custom ACT-SHAVE kernel -> NPU graph
```

ACT-SHAVE processors are the programmable part of the NPU used for software
kernels. `npunlock` compiles user C for those processors and integrates it into
a compatible graph, while Intel's existing compiler and driver remain
responsible for the graph's execution environment.

## Requirements

- Windows x64
- Meteor Lake / Intel NPU3720
- an installed Intel NPU driver for the device
- Python 3.10 or newer
- CMake 3.24 or newer and an installed MSVC toolchain for source installation
- the extracted MoviTools `MVC_DEPEND` toolchain for custom C compilation

OpenVINO is not required as a runtime, Python package, or compiler frontend.
`npunlock` does emit OpenVINO-format IR for the current Intel driver.

## Install

`npunlock` is currently installed from a source checkout:

```powershell
python -m pip install .
```

The build bundles `npunlock.dll` and `npunlock_worker.exe` inside the Python
package, so normal Python use does not require a separate native path.

## Get MoviTools

Custom C compilation uses Intel/Movidius MoviTools, which `npunlock` does not
redistribute or download. Extract the `MVC_DEPEND` payload from Lenovo's older
Intel NPU driver package `31.0.100.1688`, but **do not install or downgrade to
that driver**.

Keep the two roles separate:

```text
installed Intel NPU driver
  -> graph compilation, Level Zero, and hardware execution

extracted Lenovo 31.0.100.1688 package
  -> MoviTools compiler files only
  -> not installed
```

See [Getting MoviTools](wiki/GET_MOVITOOLS.md) for the official download,
hash, extraction command, and expected layout.

## Run an example

Point `npunlock` at the extracted `MVC_DEPEND` root and run GELU:

```powershell
$env:NPUNLOCK_MOVITOOLS_DIR = 'C:\path\to\MVC_DEPEND'
python examples\example_gelu.py
```

The example runs on the NPU and reports its maximum error against a NumPy
reference.

## What currently works

- compile user-written C into ACT-SHAVE machine code
- run custom kernels inside Intel NPU graphs
- static dense FP16 unary and two-input custom kernels
- a verified unary FP32 path
- nonlinear math such as GELU and `tanhf`
- Python, CLI, and native C APIs

## Current limitations

Support is experimental and currently limited to Windows x64, Meteor Lake /
NPU3720, static shapes, compatible ACT carriers, and known tensor layouts.
Other NPU generations have not been verified. See
[Current limitations](wiki/LIMITATIONS.md) for the full compatibility boundary.

## Documentation

- **[Getting MoviTools](wiki/GET_MOVITOOLS.md)** — obtain the compiler toolchain without installing the legacy driver
- **[Python API](wiki/PYTHON_API.md)** — construct, compile, and execute graphs from Python
- **[Writing custom kernels](wiki/CUSTOM_KERNELS.md)** — C entry point, tensor contract, and examples
- **[How npunlock works](wiki/HOW_NPUNLOCK_WORKS.md)** — graph compilation and custom-kernel integration
- **[From model to machine code](wiki/MODEL_TO_MACHINE_CODE.md)** — step-by-step lowering and the components involved
- **[Intel NPU architecture](wiki/INTEL_NPU_ARCHITECTURE.md)** — DPU and ACT-SHAVE overview
- **[Current limitations](wiki/LIMITATIONS.md)** — verified hardware and ABI scope
- **[Development and native APIs](wiki/DEVELOPMENT.md)** — CMake, testing, packaging, CLI, and C interfaces
- **[Full documentation index](wiki/README.md)**

## License

`npunlock` is licensed under the [Apache License 2.0](LICENSE). MoviTools and
the Intel/Movidius libraries are external proprietary dependencies and are not
covered or redistributed by this repository.
