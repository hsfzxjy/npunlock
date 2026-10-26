# npunlock

Intel NPUs contain programmable ACT-SHAVE cores, but the public software stack
does not expose a workflow for supplying them with custom C kernels. `npunlock`
fills that gap: it compiles C, places the resulting machine code in a validated
NPU graph, and executes the graph through Intel's driver.

![How npunlock adds custom C kernels to an Intel NPU graph](docs/npunlock-intro.svg)

The project is experimental and currently verified on Windows x64 with Meteor
Lake / NPU3720. New users should start with the
[custom-kernel tutorial](wiki/TUTORIAL.md).

## Real-world demonstrations

These contributions use custom ACT-SHAVE code beyond what the tested stock
OpenVINO paths provided. They are focused capability proofs, not claims of
universal model support or automatic speedups.

- **[YOLO11n NMS](contrib/yolo11-nms/README.md) — one native NPU graph.** The
  recorded OpenVINO stack runs the official embedded-NMS export on CPU but
  rejects it for the NPU. npunlock combines YOLO, candidate selection, packing,
  and class-aware NMS in one submission, matching the reference detections on
  two real images. The path is limited to batch 1, 255 candidates, and the exact
  reviewed model and driver/compiler hashes.
- **[XGBoost inference](contrib/xgboost/README.md) — real tree traversal on
  ACT-SHAVE.** OpenVINO 2026.3.1 cannot import the exported
  `TreeEnsembleClassifier`; npunlock instead runs branches, missing-value
  routing, accumulation, sigmoid, and classification on the NPU. All 143
  validation samples matched XGBoost's classes. Support remains limited to
  bounded numerical binary `gbtree` models.
- **[HNSW search](contrib/hnsw/README.md) — irregular graph search in a custom
  kernel.** Instead of expressing the traversal as an OpenVINO operator graph,
  the NPU performs greedy descent, dynamic neighbor expansion, visited tracking,
  and heap maintenance. All 24 recorded queries matched the host traversal
  exactly, but the demo is capped at 48 nodes and optimized CPU hnswlib is still
  much faster.
- **[Mamba-1 inference](contrib/mamba/README.md) — persistent selective state.**
  The complete Mamba-130m neural path runs on ACT-SHAVE while explicit state is
  carried across prefill chunks and decoding calls. It matched official greedy
  generation; the CPU handles tokenization, packing, submission orchestration,
  and greedy selection rather than neural layers.

Each example keeps its extra Python dependencies under `contrib/`; the core
npunlock package remains OpenVINO-free.

## How it works

`npunlock` emits OpenVINO-format IR and weights in memory and asks the Intel NPU
user-mode driver to compile a carrier graph. It separately compiles the custom C
source into a linked `3720xx` SHAVE ELF with caller-supplied MoviTools. After
validating both artifacts and the selected ACT tensor contract, it substitutes
the kernel's executable image into the native graph and runs that graph through
Level Zero.

OpenVINO-format IR is only the serialization accepted by the driver. The core
project does not import, load, link, or require OpenVINO.

## Quick start

### Requirements

- Windows x64
- Meteor Lake / Intel NPU3720 with an installed Intel NPU driver
- Python 3.10 or newer
- CMake 3.24 or newer and MSVC for installation from source
- an extracted MoviTools `MVC_DEPEND` directory to compile new custom kernels

Already-built `.npunlock` program bundles can run without MoviTools, provided
the machine has a compatible NPU driver and device.

### Install

Install from a source checkout:

```powershell
python -m pip install .
```

The installed package contains `npunlock.dll` and `npunlock_worker.exe`; normal
Python use does not require a separate native-library path.

### Get MoviTools

`npunlock` does not download or redistribute Intel/Movidius MoviTools. A
verified copy of the required `MVC_DEPEND` payload is present in Lenovo's older
Intel NPU driver package `31.0.100.1688`.

Follow [Getting MoviTools](wiki/GET_MOVITOOLS.md) for the official download,
hash, and extraction steps. Extract the toolchain only—do **not** install or
downgrade to that legacy driver.

### Verify and run

Point `npunlock` at the extracted root and inspect the local setup:

```powershell
$env:NPUNLOCK_MOVITOOLS_DIR = 'C:\path\to\MVC_DEPEND'
python -m npunlock doctor
```

The ordinary diagnostic checks the installed runtime, configured MoviTools,
and NPU/compiler availability without compiling or executing a workload. Use
`python -m npunlock doctor --full` for an explicit end-to-end compile, NPU run,
and exact host-oracle check.

Then run the bundled GELU example:

```powershell
python examples\example_gelu.py
```

## Write a custom kernel

This FP32 GELU example embeds C in Python, places it in an NPU graph, and checks
the output against NumPy. The bundled `npunlock/npu3720_kernel.h` header supplies
the invocation and tensor-address helpers. MoviTools' math archive supplies the
execution-tested `tanhf` implementation used here.

```python
import numpy as np
import npunlock as npu

npu.configure(movi_dll_dir=r"C:\path\to\MVC_DEPEND")

gelu_c: bytes = b"""
#define MLIBM_DEFINE_LINK_COMPAT 1
#include <npunlock/npu3720_kernel.h>

void controlled_act(unsigned layerParams) {
    act_abi_invocation invocation;
    ACT_ABI_LOAD_INVOCATION32_OR_RETURN(layerParams, invocation);
    const float *in = ACT_ABI_INPUT_PTR32(const float, invocation, 0u);
    float *out = ACT_ABI_OUTPUT_PTR32(float, invocation, 1u);
    const float SQRT_2_DIV_PI = 0.7978845608028654f;
    for (unsigned i = 0; i < invocation.element_count; ++i) {
        float x = in[i];
        float w = x + 0.044715f * x * x * x;
        w = w * SQRT_2_DIV_PI;
        w = tanhf(w);
        out[i] = 0.5f * x * (1.0f + w);
    }
}
"""

N = 2048
x = npu.input("x", shape=(1, N), dtype="f32")
y = npu.custom(x, source=gelu_c, carrier="Abs", _name="y")
program = npu.compile(npu.Graph(inputs=[x], outputs=[y], name="gelu_f32_example"))

input_value = np.linspace(-4, 4, N, dtype=np.float32).reshape(1, -1)
output = program.run_one(input_value)
reference = 0.5 * input_value * (
    1.0
    + np.tanh(
        np.sqrt(2.0 / np.pi)
        * (input_value + 0.044715 * input_value**3)
    )
)
print(f"maximum absolute error: {np.max(np.abs(output - reference)):g}")
```

The complete script is [examples/example_gelu_f32.py](examples/example_gelu_f32.py).
Other useful starting points include:

- [a five-input kernel mixing two tensors and global parameters](examples/example_global_hybrid_five_input_custom.py)
- [per-row scalar custom kernels](examples/example_per_row_scalar_custom.py)
- [a mixed DPU/custom-kernel network](examples/example_branched_network.py)

See the [`mlibm.a` symbol inventory](ABI/MLIBM_SYMBOLS.md) before using another
math function; an exported symbol is not considered supported until it has run
successfully on hardware.

## What works today

You can compile a C kernel to ACT-SHAVE machine code, bind it to a compatible
carrier, and run it from Python or the native C API. Hardware-tested examples
cover unary and multi-input kernels, FP16 and FP32, selected integer and
conversion paths, nonlinear math, local reductions, and bounded control flow.

Custom kernels can coexist with ordinary graph operations and DPU work.
Programs can be saved and loaded without their original C source, and tensor
I/O supports ordinary copies or reusable NumPy-compatible shared buffers.

The boundaries are intentionally strict. Shapes are static, every custom node
must match a validated carrier/tensor layout, and graph compiler 8.3 on NPU3720
is the established platform. Similar-looking graph operations are not assumed
compatible. Linux and newer Intel NPUs remain test targets rather than
supported platforms.

See [Current limitations](wiki/LIMITATIONS.md) for the exact hardware, dtype,
shape, carrier, and graph-operation matrix.

## Help test other systems

Contributions from NPU3720 Linux systems and newer Intel NPUs are welcome. The
repository includes offline blob inspection and a staged Linux execution probe,
but Linux execution and newer devices have not yet been validated. Please
record device, driver, firmware, graph-compiler, artifact hashes, and host-oracle
results rather than inferring support from graph creation alone.

See [Porting to Linux and newer NPUs](wiki/PORTING.md) and the
[porting probe bundle](wiki/PORTING_PROBE_BUNDLE.md) for the test plan.

## Documentation

- **[Custom-kernel tutorial](wiki/TUTORIAL.md)** — build a first kernel and
  progress to multi-input and mixed-precision graphs
- **[Python API](wiki/PYTHON_API.md)** — construct, compile, save, load, and execute graphs
- **[Writing custom kernels](wiki/CUSTOM_KERNELS.md)** — C entry point, tensor contract, and programming model
- **[How npunlock works](wiki/HOW_NPUNLOCK_WORKS.md)** — graph compilation and custom-kernel integration
- **[Current limitations](wiki/LIMITATIONS.md)** — exact verified hardware and ABI scope
- **[Reverse-engineering breakthroughs](wiki/REVERSE_ENGINEERING.md)** —
  experiments that established the current contracts
- **[Binary artifact visualizer](docs/VISUALIZER.md)** — inspect ELF structure, bytes, semantic spans, and references
- **[Development and native APIs](wiki/DEVELOPMENT.md)** — build, test, package, CLI, and C API guidance
- **[Full documentation index](wiki/README.md)**

> [!WARNING]
> **A note on AI use:** I did use AI while building this project--for scaffolding, repetitive implementation work, converting my reverse-engineered results into organized documentation, and fixing my English. The reverse engineering, experiments, debugging, and technical conclusions came from hands-on work. If that doesn’t bother you, there’s a pretty deep and surprisingly satisfying rabbit hole ahead.

## License

`npunlock` is licensed under the [Apache License 2.0](LICENSE). MoviTools and
the Intel/Movidius libraries are external proprietary dependencies and are not
covered or redistributed by this repository.
