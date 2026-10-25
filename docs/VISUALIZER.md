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
- `Field` optionally identifies its exact byte spans and has an optional
  plain-text description. The inspector reports the
  total annotated byte size for each field and uses a stable, distinct field
  color in both the field row and selected entity's hex bytes. Fields without
  exact spans explicitly show that their size is unknown.
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

## npunlock ELF demonstration

`build_npunlock_graph(native, kernel=None)` adds bounded ELF32/ELF64 structure
annotations and the documented NPU graph relationships. It shows ELF headers,
section records, ACT range and invocation records, their relocations, and
range-selected `KernelText` images. When a standalone kernel `.text` is exactly
byte-identical to a selected native code image, it adds that cross-artifact
relationship; it does not infer one from names or sizes.

The adapter supplies concise explanations for standard ELF regions and for
documented ACT fields. Wording explicitly distinguishes confirmed meanings
from execution-tested inferences and leaves undecoded carrier-specific fields
qualified rather than guessing.

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
