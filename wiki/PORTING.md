# Porting to Linux and newer NPUs

[Documentation index](README.md)

`npunlock` is currently verified only on Windows x64 with Meteor Lake /
NPU3720. This page describes promising routes beyond that baseline. They are
porting hypotheses and test plans, not supported configurations.

It helps to separate three compatibility questions:

1. Can the NPU execute the compiled SHAVE instructions and use the same kernel
   invocation ABI?
2. Will the driver and firmware accept the patched native graph blob?
3. Can the host run the graph compiler and MoviTools, and load the graph for
   in-process execution?

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

The version 20 JSON report records the blob size and SHA-256 plus every
discovered ACT group, invocation/range index, arity, output size, and each
tensor record's role, FP16/FP32/U8/U16/U32/I32/I16/I8 precision, element count, byte
span, and observed flags. It also reports the validated raw invocation tile
candidate, KernelParams base, and each tensor's CMX relocation symbol/addend,
rank, dimensions, and bit strides. Raw wait/post barrier masks, barrier
group/mask bytes, and start/clean scheduling values are included as well.
These routing and scheduling fields are observations, not logical tensor
offsets, semantic graph edges, or a partition/replica classification. The report can
therefore inspect the supported unary
FP16/FP32 and U8-to-FP32 conversion carriers, unary I32/I16/I8 carriers, the
equal-span binary I32/I16/I8/U16/U32 carriers, either-position scalar
FP16/FP32/U16/U32 carriers, and uniform float groups. Each target reports
`scalar_input_indices` explicitly; `input_1_scalar` remains for compatibility.

The report also lists validated network-output DMA routes: the raw CMX source,
the destination output symbol/addend, and the transfer length. For an ACT
output fully contained in one of those source intervals, the report emits an
`output_slice_candidates` entry translating the ACT subrange to a candidate
graph-output byte interval. It retains every matching route and its ambiguity
count. This matched the saved per-invocation hardware effects for four
controlled unary carriers, including the two-tile cases where only tile-0
invocations were host-visible. An invocation without such a route remains
merely unrouted; the inspector does not call it a replica.

The report also includes an `output_cover_observations` verdict for each graph
output symbol reached directly from ACT. A cover is exact only when every ACT
slice has one route, neither the route destinations nor the slices overlap,
and both merged byte-interval unions are identical. The report keeps the
number of participating ACT groups and the routed/unrouted invocation counts.
This allows a four-group per-row lowering to prove complete output coverage
without claiming that the groups correspond to one source node or that the
unrouted invocations are replicas.

The report adds a guarded replica correlation for groups containing both routed
and unrouted invocations. It does not hard-code a tile stride. Instead, it
requires the complete tensor contract, shape/stride pattern, runtime symbols,
and scheduling fields to agree, then searches for a single CMX translation
that creates a one-to-one mapping between every routed and unrouted invocation
in the group. Only one complete solution emits `replica_candidates`; zero or
multiple solutions remain unresolved. The candidate inherits the routed
graph-output interval as a proposed replica-class identity, not as permission
for automatic patch selection.

Schema v14 combines these proofs in `group_completeness_observations`. A
direct-output ACT group is complete only when every invocation has exactly one
disposition—a unique routed slice or a unique replica candidate—and output
coverage is exact. Coverage may be established by an exact partition across
the participating groups or by one group's routed slices having the same
byte-interval union as the graph-output DMA routes. The latter group-local
proof matters because a later operation may reuse the same CMX addresses; that
reuse remains visible in the conservative cross-group
`output_cover_observations` report but does not by itself invalidate an exact
group. Routed, replica, and unresolved counts remain visible per group.
Completeness still does not identify a symbolic source node and is not used by
automatic patch selection.

Schema v15 adds a validated `code_observation` to every ACT invocation. It
contains the relocated offset and extent within `.text.KernelText` plus the
SHA-256 of exactly that bounded code slice. Equal fingerprints show code reuse;
a differential compile can use changed fingerprints as evidence only when the
rest of the invocation, routing, scheduling, and DPU contract is unchanged.

Schema v16 adds `act_dpu_output_cover_observations` for intermediate ACT
groups whose outputs feed DPU inputs. A result is exact only when every
invocation output has one unique ACT-to-DPU candidate and the non-overlapping
ACT byte slices exactly cover every participating DPU input interval. DPU
endpoints are qualified by both section and invariant index because invariant
indices are not globally unique. This proves a bounded native interval and
schedule relationship; it does not establish a source-level semantic edge,
classify tile replicas, or permit automatic patch selection.

Schema v17 adds `act_consumer_cover_observations` for whole-tensor ACT-to-ACT
relationships. A producer group is exact only when every invocation output
has one unique consumer input and every such input has one candidate producer.
Fan-out, multiple possible producers, duplicated inputs, and missing producer
invocations remain unresolved. This is still native interval/schedule evidence,
not proof that either group corresponds to a particular source operation.

Schema v18 adds a raw, validated observation for every NPU3720 DMA task. Each
entry identifies its DMA engine and task index, transfer length, type-specific
geometry, barrier schedule, and source and destination relocation provenance.
Endpoints are classified conservatively as CMX, intermediate DDR buffer,
constant, graph input, graph output, another runtime symbol, or unknown. This
makes paths through intermediate DDR storage inspectable without guessing from
address values. It is groundwork for later ACT-to-DMA-to-DPU correlation, not
proof that a sequence of transfers represents a particular source-level edge.

Schema v19 adds `act_source_candidates` and `act_fanin_observations`. A source
candidate requires one ACT output to match a contiguous CMX-to-DDR task by
symbol, exact byte interval, and a direct post/wait barrier intersection. A
fan-in is exact only when every invocation in the ACT group has one unique
source task, their non-overlapping DDR destinations have one exact interval
union, and one uniquely scheduled DDR-to-multicast-CMX task consumes that
union. The result proves the bounded native data-movement chain; it does not
infer what consumes the multicast destination or identify a source node.

Schema v20 adds `act_dma_dpu_input_candidates` and
`act_dma_dpu_input_cover_observations`. For an exact fan-in group with one ACT
invocation per observed tile, the inspector derives each tile's CMX
translation from the corresponding ACT source placement relative to tile 0.
It applies that observed translation to the multicast destination and requires
one same-cluster DPU input with the translated address, equal byte span, and a
direct fan-in-post/DPU-wait barrier dependency. A cover is exact only when all
tiles form a one-to-one mapping. This is a native address-and-schedule proof;
it still does not identify the source-level operation or authorize automatic
patch selection. The public C API exposes it separately through
`patchblob_inspect_dma_dpu_group_coverage()`, and Python reports the distinct
`dma_dpu_consumer` coverage kind.

Schema v21 accepts one additional compiler-8.3 structure: several CMX
relocations may target the same DPU input register. The report preserves every
entry and sets `input_ambiguous_fields_raw` for the repeated field. It does not
choose the first or last addend. ACT/DPU and DMA/DPU derivations ignore all
relocations for that field, but may still use another unique input register in
the same invariant. Strict comparison includes the complete retained
relocation sequence and ambiguity mask, so a change cannot disappear merely
because the field is excluded from routing proofs. Output-field duplicates and
more than eight input relocations per invariant remain unsupported.

Schema v22 adds `act_dma_act_input_candidates` and
`act_dma_act_input_cover_observations`. It follows a unique ACT-output spill
into an intermediate DDR interval and requires scheduled DDR-to-CMX reloads to
match unique inputs of a later ACT group. The reload intervals must be
non-overlapping and exactly cover the spill, while unmatched producer
invocations must be unique full-contract tile replicas. Precision, byte span,
runtime symbols, addresses, and both barrier hops are checked. The public C API
exposes this proof separately through
`patchblob_inspect_dma_act_group_coverage()`, and Python reports the distinct
`dma_act_consumer` coverage kind. It does not infer a source-node identity or
a general DDR-buffer lifetime.

Schema v23 adds `dpu_act_input_candidates` and
`dpu_act_input_cover_observations` for the opposite side of a DPU region. The
current exact form requires one full-contract ACT invocation per observed tile.
It derives every tile translation from the ACT descriptors, applies that
translation to a common DPU output base, and requires one same-cluster DPU
output with equal element/byte extent and a direct post/wait dependency. The
cover remains native topology evidence only; it is not exposed as a reviewed
binding coverage kind and does not identify a source operation.

To apply that strict comparison without a custom parser, inspect a controlled
baseline and one-perturbation candidate together:

```bash
npunlock-inspect --graph baseline.blob --compare candidate.blob \
  --report comparison.json
```

The `npunlock.porting.compare.v1` result is accepted only when exactly one ACT
group changes code and every other validated invocation, tensor, route,
schedule, output, DPU, and derived-relation observation remains equal. It names
the mismatch classes when this fails. An accepted comparison is useful
differential evidence, but it still does not by itself prove a symbolic source
identity or authorize automatic patch selection.

Python callers can apply the same in-memory comparison without writing either
blob to disk:

```python
comparison = npu.compare_native_graphs(baseline_blob, candidate_blob)
print(comparison.accepted, comparison.changed_group_indices)
print(comparison.mismatch_names)
```

The native C equivalent is `patchblob_compare_graphs()`. Both APIs apply the
same fail-closed parser and comparison rules as `npunlock-inspect`.

When the baseline comes from `npu.prepare()`, Python can also produce one
controlled candidate directly:

```python
prepared = npu.prepare(graph)
probe = prepared.probe_carrier(custom_output, "Sqrt")
print(probe.accepted, probe.group_index)
print(probe.comparison.mismatch_names)
```

`probe_carrier()` changes only that custom node's serialized carrier operator,
preserves its existing inputs, outputs, attributes, weights, and build flags,
and compiles through the same native library instance. It rejects different
driver/compiler provenance before comparing. The original graph and prepared
blob are not modified. Operators that need different attributes or contracts
are not synthesized; their candidate compile may fail or the strict comparison
may reject them. A successful probe remains evidence for review, not an
automatic binding.

For stronger evidence, request two to eight independent substitutions:

```python
evidence = prepared.probe_mapping(custom_output, ("Sqrt", "Tanh"))
print(evidence.explain())
```

The status is `consistent` only when at least two accepted probes name the same
group. Accepted probes that name different groups produce `conflicting`; fewer
than two accepted probes produce `insufficient`. Rejected candidates remain in
the report. Even `consistent` evidence does not update `PreparedGraph.mappings`
or authorize automatic patch selection.

Source-label evidence and invocation coverage are separate questions. Inspect
the baseline carrier's direct-output proof explicitly:

```python
for group in prepared.inspect_group_completeness():
    print(group.group_index, group.state, group.unresolved_names)
```

Every ACT group is returned. `complete` means all invocations contributing to
one directly routed graph output are classified as a unique routed slice or a
unique tile replica and jointly have exact output coverage. `incomplete` means
that proof was attempted but unresolved. `not_observed` means the group has no
validated direct graph-output slice—commonly because it is intermediate—and
must not be read as either success or failure. This result remains diagnostic;
it does not make intermediate-group coverage claims.

Use `prepared.inspect_group_coverage()` to see all five proof kinds together
without merging their verdicts:

```python
for group in prepared.inspect_group_coverage():
    print(group.group_index, group.observed_kinds, group.complete_kinds)
```

Every group reports independent `direct_output_state`, `act_consumer_state`,
`dpu_consumer_state`, `dma_dpu_consumer_state`, and
`dma_act_consumer_state` values plus route-specific counts and unresolved
flags. The last two are obtained from their separate C inspection functions
and merged only as a Python diagnostic view. Neither API chooses which
evidence kind is sufficient for a binding, and neither combines coverage with
a symbolic source label.

Python callers can make that policy choice explicitly after inspecting both
reports:

```python
reviewed = prepared.review_binding(
    custom_output,
    evidence,
    coverage_kind="act_consumer",
)
plan = prepared.plan(bindings={custom_output: reviewed})
```

This accepts only `consistent` multi-probe evidence and a caller-selected
coverage kind whose state is `complete`. The immutable result records the
prepared native-blob hash, group, evidence, and coverage; `plan()` revalidates
them before resolving the existing explicit group binding. It is a review aid,
not an automatic source mapper or an unforgeable authorization token: callers
can still bind an `ActGroup` or integer index directly when they have external
evidence.

The same report includes the exact compiler-8.3/NPU3720 DPU invariant form.
For each 304-byte invariant
it reports raw wait/post scheduling fields and the CMX runtime relocations for
the architectural input-address and output-multicast register fields. It also
decodes the raw input/output X/Y/Z and dtype fields and derives checked element
and byte extents for static dense byte-addressable tensors. Packed or sparse
forms remain unsupported instead of receiving a guessed physical extent. This
can show a concrete ACT-output -> DPU-input and DPU-output -> ACT-input interval
chain in a controlled graph.

These observations are still not a complete route from every native task to a
symbolic source tensor. Only the first two of the four architectural input and
output address fields have appeared in the retained fixtures so far. Raw DMA
tasks now expose internal transfer endpoints, extents, and schedules, but
address reuse and ambiguous chains remain unresolved. The inspector does not
promote a matching address/schedule chain into a semantic edge or automatic
patch target.

The `act_dpu_edge_candidates` list performs the next bounded correlation. An
ACT tensor must be fully contained in a DPU dense CMX interval with the same
runtime symbol, and the producer's post mask must directly intersect the
consumer's wait mask. Identical DPU register relocations are represented once
with a raw field mask. Every candidate includes the count for its ACT tensor;
non-unique matches remain visibly ambiguous. These are native interval and
schedule candidates, not semantic graph edges and not permission to select a
patch target.
That includes the documented equal-span binary FP32 `Subtract` carrier.
It also recognizes the structurally identical dense FP32 `(8,8)->8`
invocations emitted for the documented row-broadcast carrier; inspection alone
does not recover or prove its source-level broadcast shape.

The `act_edge_candidates` list makes that correlation explicit when an earlier
ACT output and a later ACT input have the same CMX symbol, exact byte interval,
precision, and element count, and a forward path exists through the observed
ACT post/wait barrier masks. `schedule_hop_count` is one for a direct barrier
match and greater than one for a transitive ACT-only scheduling path. The
direct producer/consumer mask intersection remains available as a raw field
and is zero for a purely transitive match.
`consumer_candidate_count` is reported on every match. Only a count of one is
unique within this narrow correlation; any other count is deliberately left
ambiguous. Even a unique candidate is diagnostic evidence, not a source-level
graph edge or permission to select a patch target automatically. Barrier
ordering can serialize tasks without proving data flow, and address reuse can
still create false correlations.
Discovery means only that the
file matches the narrow observed NPU3720/compiler-8.3 structures. It does not
prove that a Linux driver, newer firmware, or different NPU can load or execute
the blob.

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

### Milestone 4: POSIX compiler-worker isolation

A reusable bounded POSIX process launcher is now implemented and tested
offline. It uses a dedicated binary-response descriptor, nonblocking request
transport, separate stdout and stderr pipes, a child process group, finite
deadlines, and deterministic group termination. The focused fixture verifies
both byte-exact transport and timeout cleanup.

The launcher is reserved for future graph-compilation and possible
Wine-hosted MoviTools workers. Inference intentionally remains in-process on
both platforms, with callers choosing copied tensors or explicit host/NPU
shared allocations.

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

- **Run an existing patched blob.** Use the Linux Level Zero graph loader and
  in-process execution probe, then test a blob produced and patched on Windows.
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
