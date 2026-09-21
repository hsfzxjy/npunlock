# npunlock

**Custom C kernels for Intel Core Ultra NPUs.**

`npunlock` lets you place user-written C computation inside an Intel NPU graph
and run it on the NPU's programmable ACT-SHAVE processors. Intel's normal
software stack exposes the NPU at the model level; `npunlock` adds a narrow,
validated path for custom kernels that the graph API does not otherwise expose.

The project is experimental and currently targets Windows x64 with Meteor Lake
/ NPU3720. It does not require OpenVINO as an application runtime or Python
dependency. It does use OpenVINO-format IR when talking to the installed Intel
NPU driver, which remains responsible for graph compilation and hardware
execution.

```text
Python graph + C kernel
          |
          v
       npunlock
          |
          v
 current Intel NPU driver
          |
          v
         NPU
```

## Why npunlock?

Model compilers can run only the operations and combinations they know how to
lower. When an operation is missing, unusually specialized, or better expressed
as a small C routine, the usual choices are to reformulate the model or move
that work off the NPU.

`npunlock` provides another option: keep ordinary layers in the normal Intel
graph pipeline while supplying C for selected programmable operations. It uses
Intel's compiler and current driver to construct and execute the graph, so it
does not try to replace the NPU compiler or use programmable code in place of
the dedicated tensor engines.

## Quick example

From the repository root, this example creates an FP16 graph whose custom
operation runs a C GELU kernel on ACT-SHAVE. The full kernel source lives in the
runnable example file, keeping the graph code small:

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

program = npu.compile(npu.Graph([x], [y], name="gelu_example"))
input_value = np.linspace(-4, 4, 2048, dtype=np.float16).reshape(1, -1)
output = program.run({"x": input_value})["gelu"]
```

Ordinary graph nodes are serialized for the Intel compiler as usual. A custom
node carries user C; `npunlock` compiles it, integrates it with a compatible
ACT operation, and executes the resulting graph through the current NPU driver.

See the complete [FP16 GELU example](examples/example_gelu.py), the
[FP32 GELU example](examples/example_gelu_f32.py), and the
[multi-layer two-input example](examples/example_multilayer_multi_input.py).

## Requirements

- Windows x64
- Meteor Lake / Intel NPU3720 (the currently confirmed hardware)
- a current Intel NPU driver
- Python 3.10 or newer
- CMake 3.24 or newer and an installed MSVC toolchain for source installation
- the extracted MoviTools `MVC_DEPEND` toolchain for custom C compilation

OpenVINO is not required.

## Install

`npunlock` is currently installed from a source checkout; no PyPI release is
documented yet. From the repository root:

```powershell
python -m pip install .
```

The build compiles and bundles `npunlock.dll` and `npunlock_worker.exe` inside
the Python package. Normal Python use does not require a separate native path.

For native C development and offline tests, see
[Development](wiki/DEVELOPMENT.md).

## Get MoviTools

Custom C compilation currently relies on Intel/Movidius MoviTools. `npunlock`
does not redistribute or automatically download these proprietary files.

The known-good tools can be extracted from Lenovo's original Intel NPU driver
package version `31.0.100.1688`. Do **not** install or downgrade to that driver;
only extract its `MVC_DEPEND` compiler payload. Keep the current Intel NPU
driver installed for graph compilation and execution.

See **[Getting MoviTools](wiki/GET_MOVITOOLS.md)** for the official Lenovo
links, exact SHA-256, tested extraction command, and expected directory layout.

## Run an example

After extracting MoviTools, set its root for the current PowerShell session and
run the FP16 GELU example:

```powershell
$env:NPUNLOCK_MOVITOOLS_DIR = 'C:\path\to\MVC_DEPEND'
python examples\example_gelu.py
```

The example compiles the graph and C kernel, executes them on the NPU, and
prints the maximum error against a NumPy reference.

## What currently works

- C source to a validated `3720xx` SHAVE executable
- integration with an Intel-driver-compiled native NPU graph
- bounded graph execution with NumPy inputs and outputs
- static dense FP16 unary and validated two-input custom kernels
- nonlinear kernels, including GELU with MoviTools math code
- a narrow precision-preserved unary FP32 path
- automatic positional selection for unambiguous one-to-one ACT graphs
- Python, CLI, and independent C17 component APIs

## Current limitations

The validated scope is intentionally narrow: Windows x64, Meteor Lake /
NPU3720, static shapes, compatible ACT carriers, and observed dense tensor
layouts. Dynamic shapes, broadcasting, arbitrary layouts and precisions,
arbitrary graph-to-ACT mapping, and other NPU generations are not currently
supported. Ambiguous or unfamiliar native graphs are rejected rather than
patched speculatively.

See [Current limitations](wiki/LIMITATIONS.md) for the detailed boundary.

## Documentation

The [documentation index](wiki/README.md) links all technical guides. Start
with whichever question matches what you need:

### How do I obtain MoviTools safely?

Extract the compiler payload from the known Lenovo package without installing
its legacy driver. [Getting MoviTools](wiki/GET_MOVITOOLS.md) provides the
official links, expected hash, extraction command, layout, and provenance
notes.

### How do I construct and run a graph from Python?

The Python frontend builds symbolic graphs, accepts custom C operations,
compiles them, and runs NumPy tensors through the bundled native runtime. See
the [Python API guide](wiki/PYTHON_API.md) for the actual graph, configuration,
compilation, and execution interfaces.

### What does a custom C kernel look like?

Custom kernels use a narrow C entry-point and tensor-descriptor contract rather
than an unrestricted hosted C environment. [Writing custom kernels](wiki/CUSTOM_KERNELS.md)
documents the entry point, observed tensor layouts, supported precisions,
multi-input behavior, math linkage, and examples.

### How does npunlock put custom code into an NPU graph?

Intel's compiler first creates a valid graph around a compatible carrier
operation; `npunlock` then substitutes the selected ACT-SHAVE executable while
preserving that execution environment. [How npunlock works](wiki/HOW_NPUNLOCK_WORKS.md)
explains the complete pipeline and separates confirmed behavior from current
inferences.

### What are DPU and ACT-SHAVE processors?

The DPU handles regular tensor computation, while ACT-SHAVE provides the
programmable path used by custom kernels. The
[Intel NPU architecture overview](wiki/INTEL_NPU_ARCHITECTURE.md) introduces
those roles without assuming prior NPU knowledge.

### Which hardware, shapes, and data types are supported?

The confirmed contract is intentionally smaller than the capabilities of the
underlying hardware. [Current limitations](wiki/LIMITATIONS.md) lists the
validated platform, tensor, carrier, mapping, ELF, and graph-patching boundaries.

### Can I use the native libraries without Python?

Yes. The project exposes separate C17 interfaces for kernel compilation, graph
compilation, patching, and execution. The [C API guide](docs/C_API.md) covers
ownership, deployment, diagnostics, and component-level use; the detailed
binary observations remain under [`ABI/`](ABI/).

### How do I build, test, or contribute to the native project?

The [development guide](wiki/DEVELOPMENT.md) describes the CMake presets,
native targets, unified worker, CLI, package layout, offline tests, and opt-in
hardware tests.

## License

`npunlock` is licensed under the [Apache License 2.0](LICENSE). MoviTools and
the Intel/Movidius libraries are external proprietary dependencies and are not
covered or redistributed by this repository.
