# npunlock

`npunlock` is a narrow Windows MVP for compiling C into a validated 3720xx
SHAVE ELF, compiling OpenVINO-format IR XML/BIN into an Intel NPU native graph
blob without OpenVINO, and replacing explicitly selected compatible ACT kernel
ranges.

The supported MVP contract is intentionally limited to the experimentally
confirmed static dense FP16 carrier class. It does not infer graph source-node
names, dynamic shapes, broadcasting, arbitrary layouts, or a universal ACT ABI.

## Current state

The standalone C17 project and public C ABI are scaffolded. The three library
entry points currently validate their public arguments and return a structured
`not_implemented` diagnostic until their respective implementation milestones
land.

Configure, build, and run the offline tests on Windows:

```powershell
cmake --preset windows-vs2022
cmake --build --preset windows-debug
ctest --preset windows-debug
```

Hardware tests remain disabled unless `NPUNLOCK_ENABLE_NPU_TESTS=ON` is set
explicitly. OEM Movi DLLs are never bundled and their location will be supplied
by the caller.

See [docs/PLAN.md](docs/PLAN.md) for the implementation sequence and
[LICENSES/PROVENANCE.md](LICENSES/PROVENANCE.md) before moving any retained
research artifact into this repository.
