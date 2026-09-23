# Development

[Documentation index](README.md)

This page describes the repository's native architecture, build, tests, and
packaging. User-facing Python setup remains in the root [README](../README.md).

## Native components

Four public C17 component APIs are exported from one `npunlock.dll`:

| Component | Responsibility |
| --- | --- |
| `shavecc` | C source plus `MVC_DEPEND` root -> validated SHAVE ELF |
| `ir2blob` | OpenVINO-format IR buffers -> native graph blob |
| `patchblob` | graph plus SHAVE ELF -> validated patched graph |
| `graphinfer` | native graph plus tensor buffers -> output buffers |

The installed CMake targets remain separate:

```text
npunlock::shavecc
npunlock::ir2blob
npunlock::patchblob
npunlock::graphinfer
```

All resolve to the same shared runtime. `npurun` is the file-oriented CLI over
those buffer-oriented libraries.

## Worker isolation

MoviTools calls, Intel graph compilation, and default NPU execution can hang,
fault, or terminate their process. One `npunlock_worker.exe` dispatches all
three worker modes, but each ordinary request still runs in its own
finite-lived process with a deadline and Windows Job Object cleanup.

The worker communicates with the libraries through bounded, versioned pipe
messages. Stage buffers remain in memory; no temporary files connect native
library stages.

The three library clients share one Win32 process-launch implementation. It
owns worker discovery, inheritable handles, request writing, response draining,
stdout/stderr capture, deadlines, and Job Object cleanup. Each mode retains
only its protocol-specific request builder and response parser. The worker
executable similarly shares byte encoding, bounded input, complete output, and
response-handle helpers among its three modes.

`graphinfer` additionally exposes an opt-in in-process session for directly
binding host/NPU shared Level Zero allocations. This is required for NumPy to
view the same allocation used by the NPU. The session serializes executions and
uses finite fence waits, but it cannot provide Job Object recovery from a
driver call that never returns.

Protocol responses, stdout, and stderr are independent byte streams. The C
results expose both captured process streams. `npurun` writes both verbatim
when a worker fails and reports nonempty stderr as a warning after success.

## Full Windows runtime requirements

- Windows x64
- CMake 3.24 or newer
- an installed MSVC toolchain
- Python 3.10 or newer with NumPy for Python tests
- Black for Python formatting
- `clang-format` for the formatting targets

The public CMake preset expresses the Windows build but does not force a
particular Visual Studio generator or version.

Run MSVC-dependent commands through the repository wrapper:

```powershell
tools\msvc-run.ps1 cmake --preset windows
tools\msvc-run.ps1 cmake --build --preset windows-debug
tools\msvc-run.ps1 ctest --preset windows-debug
tools\msvc-run.ps1 cmake --build build\windows --config Debug --target format-check
python -m black --check python examples tests\python tools setup.py
```

The default test suite is offline. It does not require MoviTools, an NPU, or
OpenVINO.

## Linux porting utilities

The Linux preset builds `npunlock-inspect`, the experimental in-process
`npunlock-linux-probe`, and their offline tests. It requires a C17 compiler and
CMake 3.24 or newer. It does not build or invoke MoviTools, graph compilation,
or the Windows worker. Ordinary CTest does not attempt NPU execution:

```bash
cmake --preset linux-inspect
cmake --build --preset linux-inspect
ctest --preset linux-inspect
./build/linux-inspect/npunlock-inspect --graph patched.blob
```

The execution probe dynamically loads the system Level Zero loader only when
explicitly invoked with a verified probe bundle. The full compiler, patch, and
public execution runtime remains Windows-only. Read the
[staged Linux roadmap](PORTING.md#porting-utilities-and-staged-roadmap) before
interpreting results as driver, firmware, instruction-set, or invocation-ABI
compatibility.

CTest also exercises the private POSIX worker-process transport without an NPU.
The fixture checks the dedicated response channel, separate stdout/stderr
capture, and process-group timeout cleanup. It does not invoke Level Zero.

## Opt-in integration tests

First configure the `MVC_DEPEND` root:

```powershell
$env:NPUNLOCK_MOVITOOLS_DIR = 'C:\path\to\MVC_DEPEND'
```

Then enable the suites needed by the machine:

```powershell
tools\msvc-run.ps1 cmake --preset windows -B build\integration `
  -DNPUNLOCK_ENABLE_MOVITOOLS_TESTS=ON `
  -DNPUNLOCK_ENABLE_NPU_TESTS=ON `
  -DNPUNLOCK_ENABLE_GRAPHINFER_TESTS=ON
tools\msvc-run.ps1 cmake --build build\integration --config Debug
tools\msvc-run.ps1 ctest --test-dir build\integration -C Debug --output-on-failure
```

These tests invoke proprietary tools and/or hardware and are intentionally not
part of ordinary `ctest`.

## Install the native C package

```powershell
tools\msvc-run.ps1 cmake --install build\windows --config Debug --prefix build\stage
```

The installation contains headers, CMake package files, `npunlock.dll`,
`npunlock_worker.exe`, and `npurun.exe`. Deploy the DLL and worker together.

See the [C API guide](../docs/C_API.md) for buffer ownership, diagnostics,
release functions, and consumer linking.

## File-oriented CLI

`npurun build` is the file boundary over the four C libraries. It compiles an
IR graph, compiles a C kernel, selects a validated positional ACT group, and
writes the patched graph plus a provenance manifest:

```powershell
npurun build `
  --ir model.xml `
  --weights model.bin `
  --shave-source kernel.c `
  --movi-dll-dir C:\path\to\MVC_DEPEND `
  --patch-position 0 `
  --output patched.blob `
  --manifest patched.json
```

`--patch-position` is a zero-based discovered ACT group, not a source node
name. Explicit invocation/range and tensor-contract options exist for advanced
validated layouts. The CLI argument overrides `NPUNLOCK_MOVITOOLS_DIR`; when
the argument is omitted, the environment variable supplies the root.

The optional `--run-add1`, `--run-input`, and `--run-output` path is a narrow
FP16 add-one oracle, not a general CLI semantic validator. General tensor
execution is exposed by `graphinfer` and Python `Program.run()`.

## Build the Python package

The Python build invokes CMake and bundles `npunlock.dll` plus
`npunlock_worker.exe` inside the package:

```powershell
python -m pip install .
```

The package-local native directory is the default at runtime. Developers can
override it with `native_dir=` or `NPUNLOCK_NATIVE_DIR` when testing another
build tree.

The package is not currently documented as a PyPI release; installation is
from the checked-out source tree.

## Source layout

```text
include/npunlock/    public C headers
src/shavecc/         compiler orchestration and ELF validation
src/ir2blob/         Intel driver graph compilation
src/patchblob/       native graph validation and mutation
src/graphinfer/      graph execution API
src/workers/         isolated MoviTools/driver worker modes
src/npurun/          file-oriented CLI
python/npunlock/     symbolic Python frontend and ctypes bindings
examples/            runnable Python custom-kernel examples
tests/               offline and opt-in integration tests
ABI/                 observed binary-interface references
```

## Validation policy

External binaries are untrusted parser inputs. Every offset, extent,
relocation, count, alignment, and size calculation must be bounds- and
overflow-checked. Unsupported structures fail before hardware execution.

The four C APIs exchange only memory buffers and define matching release
functions. A buffer must be released by the runtime that allocated it. Python
copies ordinary native results before release; shared arrays instead retain
their native buffer owner for the lifetime of every NumPy view.

## Technical references

- [How npunlock works](HOW_NPUNLOCK_WORKS.md)
- [Native graph ELF ABI](../ABI/GRAPH_ELF.md)
- [ACT kernel ELF ABI](../ABI/KERNEL_ELF.md)
- [MoviTools DLL contract](../ABI/MOVITOOLS.md)
- [C API guide](../docs/C_API.md)

[Back to documentation index](README.md)
