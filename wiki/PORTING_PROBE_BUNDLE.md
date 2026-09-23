# Porting probe bundle

[Documentation index](README.md) · [Porting roadmap](PORTING.md)

The first execution probe is a fixed static dense FP16 add-one graph. Keeping
the graph, input, expected output, and provenance together prevents a test on a
new operating system or NPU from silently changing several variables at once.

This bundle is evidence for one experiment, not a general compatibility claim.

## Create a bundle

Generate the patched graph through the validated Windows path, then create the
bundle with the standard-library-only helper:

```powershell
python tools\porting_probe.py create `
  --graph patched-add1.blob `
  --output-dir build\add1-probe `
  --producer-platform "Windows 11 x64" `
  --device "Meteor Lake / NPU3720" `
  --driver-version "31.x.x.x" `
  --graph-compiler-version "8.3"
```

The destination must not already exist. The helper creates exactly four files:

| File | Purpose |
| --- | --- |
| `manifest.json` | Schema, producer provenance, tensor selectors, hashes, and oracle contract |
| `graph.blob` | Complete prepatched native graph |
| `input-0.bin` | Deterministic `1x16` little-endian FP16 input tensor |
| `expected-output-0.bin` | Exact FP16 result of adding `1.0` to every input element |

The tensor selectors are argument index `0` for the input and argument index
`1` for the output. The first bundle intentionally supports only this exact
contract.

## Verify before execution

On the destination machine, verify all hashes, the fixed tensor contract, the
file set, and the host oracle before loading the graph:

```bash
python3 tools/porting_probe.py verify build/add1-probe
./build/linux-inspect/npunlock-inspect \
  --graph build/add1-probe/graph.blob \
  --report build/add1-probe/inspect.json
```

`inspect.json` is intentionally outside the four-file bundle contract. Put it
elsewhere when running strict bundle verification, or verify before generating
the inspection report.

The verifier uses Python's IEEE-754 binary16 support and compares the expected
output bytes exactly. It does not load Level Zero and works without an NPU.

## Manifest contract

The schema identifier is `npunlock.porting.probe.v1`. Each binary entry records
its relative filename, byte count, and SHA-256. Tensor entries additionally
record argument index, `f16` dtype, and `[1, 16]` shape. Producer platform,
device, driver version, and graph-compiler version are required nonempty
strings; use an explicit `unknown` rather than omitting a value.

The bundle must contain no MoviTools DLLs, Intel driver binaries, machine-local
paths, credentials, or private model data. The four project-generated files may
be archived for transport after verification.

## How to interpret a future execution result

- Bundle verification proves only transfer integrity and host-oracle
  consistency.
- `npunlock-inspect` success proves only compatibility with the observed graph
  structures parsed by this project.
- Graph creation success proves driver acceptance, not execution.
- Execution completion proves neither correct SHAVE instructions nor a stable
  invocation ABI.
- Only an exact match with `expected-output-0.bin` establishes this one add-one
  experiment on the reported platform.

Linux graph loading and execution are the next roadmap milestone and are not
implemented by the bundle helper.
