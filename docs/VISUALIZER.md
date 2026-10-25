# Semantic binary artifact visualizer

`npunlock.viz` is a read-only, format-agnostic hex viewer. Python provides
binary artifacts plus semantic spans, hierarchy, fields, addresses, and typed
cross-references. The browser treats the supplied bytes as the source of truth
and fetches only bounded byte ranges from a loopback server.

## Smallest useful example

```python
from npunlock.viz import ArtifactGraph, Span, show

graph = ArtifactGraph("Example")
native = graph.artifact("native", data=graph_blob, name="graph.blob")
code = graph.node(
    "code", label="KernelText", kind="code", category="code",
    spans=[Span(native, 0x1000, 0x1200)],
)
record = graph.node(
    "range-0", label="ACT range 0", kind="record",
    spans=[Span(native, 0x400, 0x418)], fields={"code_size": 0x200},
)
graph.link(record, code, "uses_code")
show(graph)
```

`show()` binds only to `127.0.0.1`, opens the default browser, and blocks until
Ctrl+C. The blocking wait remains interruptible in Windows PowerShell and
closes the HTTP server before returning. Interactive code can retain an
explicit session instead:

```python
session = show(graph, block=False)
print(session.url)
# ...
session.close()
```

`ViewerSession` is also a context manager. There is no account, database,
internet request, edit endpoint, or JavaScript toolchain requirement at
runtime.

## Command line

The installed package exposes identical module and executable entry points:

```powershell
python -m npunlock.viz graph.blob
npunlock-viz graph.blob
```

Pass a standalone SHAVE ELF second to add its structure and any exact
executable-byte relationship to the same workspace:

```powershell
python -m npunlock.viz graph.blob kernel.elf
npunlock-viz graph.blob kernel.elf
```

The first path is the primary artifact and the optional second path is the
standalone kernel. Inputs are validated by ELF contents, not filename
extensions. Both forms open the default browser, print the loopback URL, and
block until Ctrl+C. `--no-browser` leaves opening the printed URL to the user;
`--port` selects a stable loopback port, and `--title` overrides the workspace
title. Use `--help` for the complete contract.

## Model

The serialization envelope is `npunlock.artifact-graph`, version 1.

- An `Artifact` has a stable ID, display name, byte size, SHA-256 digest, and
  immutable bytes or a file source. A file is range-read and rejected if its
  size or modification timestamp changes after graph construction.
- A `Span` is a nonempty file-relative half-open interval `[start, end)` and
  always identifies its artifact. It never represents a virtual address.
- A `Node` has a stable ID, label, kind, semantic category, optional JSON
  properties, an optional plain-text description, zero or more spans, fields,
  and explicitly named non-file `Address` values. Logical nodes need no byte
  span; one node may cover several noncontiguous spans.
- `Field` optionally identifies its exact byte spans and has optional
  plain-text description and JSON properties. The inspector reports the
  total annotated byte size for each field and uses a stable, distinct field
  color in both the field row and selected entity's hex bytes. Fields without
  exact spans explicitly show that their size is unknown. Descriptor-backed
  fields use properties to carry evidence confidence and the scope in which a
  meaning was established; the inspector shows those qualifications beside
  the explanation. A field or node may also set `relevance` to `relevant`,
  `context`, or `irrelevant`. Marking something irrelevant is an affirmative
  semantic claim and must not be inferred merely because its meaning is
  unknown.
- `contain(parent, child)` defines the single-parent hierarchy. `link(source,
  target, kind)` adds arbitrary typed relationships independently of that
  hierarchy. Links may cross artifacts.
- Nested and overlapping spans are valid. The hex view chooses the shortest
  covering span as the primary color, then the deeper hierarchy node. The
  inspector keeps every covering entity reachable.
- The hierarchy is fully expanded on first load so every entry is immediately
  present; branches can then be collapsed with the mouse or keyboard.

`graph.dumps(include_data=True)` creates a portable JSON document with base64
artifact data. The default omits bytes, matching the browser metadata endpoint;
`ArtifactGraph.loads(..., artifact_data={...})` can bind those metadata-only
documents back to bytes or paths. All references and bounds are validated
before serving.

Categories use one restrained color each and are configurable by passing
`Category` values to `ArtifactGraph`.

The `focus relevant` toolbar option strongly fades both the foreground and
semantic fill of hex/ASCII bytes explicitly carrying `relevance:
"irrelevant"`. It also fades the complete field row in the inspector and the
complete node or section row in the structure tree. This is a focus aid, not a
claim that muted bytes can be discarded: bytes remain the source of truth and
the viewer never hides them. The option is enabled by default and leaves the
active byte selection at full contrast. Its state, structure/inspector
visibility, and user-adjusted panel widths are stored in `localStorage`;
malformed or unavailable storage falls back safely to defaults. Browser
storage is scoped by origin, including the port, so use a stable explicit
`show(graph, port=...)` value when preferences should carry between separate
viewer processes. Random-port sessions remain isolated.

Selecting any node with physical spans adds a compact, left-facing bracket rail
just to the right of the ASCII column. The rail follows every visible row
intersected by the node and uses start/end caps for each physical span, so a
section header and a noncontiguous payload remain visibly separate while nested
records such as DMA tasks receive the same emphasis. Byte highlighting
continues to carry exact within-row boundaries.

## Format descriptors

Fixed visualization metadata is bundled under `npunlock.viz/schemas/` rather
than embedded in the parser. `elf.json` describes ELF32/ELF64 header and
section-header fields. `npu3720-act-compiler-8.3.json` describes the observed
ACT range, invocation, and Rela records for the supported carrier family.
These files own fixed record sizes, relative field offsets and encodings,
plain-text explanations, node presentation metadata, and section
classification rules. `descriptor.schema.json` documents their versioned JSON
shape.

The boundary is deliberately hybrid. Python still recognizes and bounds-checks
ELF containers, resolves dynamic section and string tables, validates record
counts, and creates relationships whose targets depend on parsed values. JSON
does not contain expressions, Python names, pointer arithmetic, or a query
language. The loader accepts only a closed set of integer encodings and rejects
unknown keys, out-of-record fields, inconsistent sizes, and unknown bundled
descriptor names before decoding bytes. This keeps untrusted artifact bytes and
untrusted external schemas from becoming executable parser logic.

The bundled descriptors are an internal adapter boundary in version 1, not a
general plugin API. Adding another reviewed binary format means adding a
descriptor for its fixed records and a small Python adapter for its structural
validation and relationships. The generic `ArtifactGraph` serialization stays
format-agnostic; optional field properties are an additive version-1 feature.

The inspector also summarizes what the graph knows about the active artifact:
the union of bytes covered by semantic regions, the union of exact decoded
field spans, entity and relationship counts, and how many fields have an
explanation. A whole-file root annotation is excluded from region coverage so
it cannot manufacture a misleading 100% result. These are annotation-coverage
figures, not claims that every covered byte has a confirmed ABI meaning.

## npunlock ELF demonstration

`build_npunlock_graph(native, kernel=None)` adds bounded ELF32/ELF64 structure
annotations and the documented NPU graph relationships. It shows ELF headers,
section records, ACT range and invocation records, their relocations, and
range-selected `KernelText` images. When a standalone kernel `.text` is exactly
byte-identical to a selected native code image, it adds that cross-artifact
relationship; it does not infer one from names or sizes.

For the supported NPU3720/compiler-8.3 carrier family, the adapter also exposes
the bytes that npunlock actually uses to accept and patch a graph:

- complete ELF headers and section headers, symbol entries, and every ELF64
  Rela entry;
- compiler/NPU/tile/loader compatibility tokens, loader and mapped-inference
  version notes, and the fixed platform-contract words;
- every byte of ACT range and invocation records, including code selection,
  parameter and kernel-data references, barrier masks, scheduling values, tile
  selection, and explicitly raw identity words;
- relocation-selected `KernelParams` blocks, their 0x28-byte `MemRefData`
  records, CMX addends, element types, ranks, dimension arrays, bit strides,
  derived element counts and spans, and dense-layout result;
- the DMA fields and DPU-invariant fields used by npunlock's routing analysis,
  with relocations linked back to the exact records they patch.

Field spans are exact file-relative byte intervals, so selecting an inspector
field highlights the bytes that encode it. Derived values such as a resolved
CMX addend, tensor element count, or transfer type are properties or addresses,
not invented file spans. Carrier bytes that npunlock preserves but does not
understand are labeled as raw instead of receiving guessed names. Large opaque
compiler metadata, runtime configuration, and unconsumed task fields remain
region-annotated rather than being presented as a supported ABI.

The adapter supplies concise explanations for standard ELF regions and for
documented ACT fields. Every descriptor-backed field carries its available
evidence label (`confirmed`, `strong inference`, `execution-tested inference`,
or `raw observation`) and any applicable carrier scope into the inspector and
byte tooltip. Wording leaves undecoded carrier-specific fields qualified rather
than guessing. The ELF root spans the complete artifact and states the parsed
class, byte order, object type, machine, section count, and file size, so the
file-level interpretation remains reachable from any byte.
Reserved ELF identification padding is explicitly marked as confirmed
irrelevant because the standard assigns it no supported meaning. Raw or
unexplained carrier fields are not dimmed: uncertainty is not irrelevance.
The npunlock adapter also marks ELF null-section sentinels, unreferenced
all-zero null-symbol sentinels, and inactive or redundant DMA record
interpretations as irrelevant. These classifications are conditional on the
parsed artifact; a referenced symbol or active DMA interpretation stays at
full emphasis.

The optional `target_groups=` argument accepts the existing
`NativeLibraries.discover_patch_targets()` or `discover_patch_targets_v2()`
result and adds the validated tensor contract to invocation properties. The
visualization parser is diagnostic only: it is not used to authorize patching.

Run either hardware-independent example from a source checkout:

```powershell
$env:PYTHONPATH = "python"
python examples\visualize_synthetic.py
python examples\visualize_npunlock.py
```

The second command uses retained test fixtures and does not access MoviTools,
the NPU driver, or hardware.

## Frontend maintenance

Compiled `app.js`, `app.css`, and `index.html` are package data. End users do
not need Node.js. To change the TypeScript source:

```powershell
cd frontend
npm install
npm run check
npm run build
```

The implementation uses no runtime JavaScript dependencies or UI framework.
