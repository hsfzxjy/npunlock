# HNSW search on ACT-SHAVE

This example runs a complete bounded HNSW query traversal inside a custom
ACT-SHAVE kernel on Intel Meteor Lake. The NPU follows graph links, computes
squared-L2 distances, maintains candidate/result heaps, tracks visited nodes,
and stops through the normal `efSearch` bound.

The CPU builds and serializes the index, prepares query batches, and checks the
answers. It does not choose or repair the NPU search path. Host verification is
enabled by default and can be disabled only after a caller has completed its
own acceptance checks.

The backend declares its ACT layout and executable tensor contract through the
public `npunlock` contract API. Its optional persistent cache is the shared
provenance- and content-bound core cache; this contribution contains no native
blob parser or cache implementation of its own.

## Try it

Create the environment from the repository root:

```powershell
python -m venv build\hnsw-venv
$python = '.\build\hnsw-venv\Scripts\python.exe'
& $python -m pip install -r contrib\hnsw\requirements.txt
& $python -m pip install .
$env:PYTHONPATH = (Resolve-Path contrib\hnsw)
```

Run the real-data comparison on the CPU first:

```powershell
& $python contrib\hnsw\scripts\run_demo.py `
  --cpu-only `
  --output-dir contrib\hnsw\outputs\cpu
```

Then run the same index and queries on the NPU:

```powershell
& $python contrib\hnsw\scripts\run_demo.py `
  --movi-dll-dir C:\path\to\MVC_DEPEND `
  --native-dir build\windows\Debug `
  --output-dir contrib\hnsw\outputs\npu
```

The demo uses real handwritten-digit data, produces 16-dimensional PCA
embeddings, builds a 48-node index, and searches 320 held-out queries. Generated
indices, compiled programs, and reports stay under ignored output directories.

## What was verified

On the recorded NPU3720/compiler-8.3 system:

- all 320 NPU searches matched the independent Python HNSW traversal exactly in
  neighbor IDs, raw FP32 distances, and traversal counters;
- the NPU visited 26.4 of 48 nodes on average, so the run was not exhaustive;
- recall@5 was 0.897 for this small `M=4`, `efSearch=16` graph;
- one 12,864-byte SHAVE image served four ACT targets; and
- a 320-query batch averaged 2.165 ms of graph execution (147,783 query/s) and
  2.428 ms for the complete public call (131,779 query/s) in the retained
  5,000-run measurement.

hnswlib processed the same tiny-index batch in 0.810 ms. This contribution shows
that irregular graph traversal works on the NPU; it does not claim a CPU
performance crossover.

## Public smoke test

The public test checks deterministic construction, search, batching, and index
serialization without an NPU or MoviTools:

```powershell
$env:PYTHONPATH = (Resolve-Path contrib\hnsw)
& $python -m unittest discover -s contrib\hnsw\tests -p 'test_hnsw.py' -v
```

## Limits

- At most 48 nodes, 32 dimensions, four graph layers, four neighbors per layer,
  `efSearch=24`, and top-k 16.
- One graph submission accepts up to 320 independent queries.
- The tested carrier is FP32 and keeps every query/index chunk within one ACT
  invocation; there is no cross-invocation communication.
- Static search metadata and the complete adjacency graph are validated once
  per invocation before its query traversals begin.
- The 16-dimensional fast path uses a fixed four-lane FP32 reduction tree; the
  host oracle reproduces that order instead of assuming scalar accumulation.
- The common `efSearch=16`, top-k 5 case has a compile-time-bounded fast path;
  other supported values continue through the configurable kernel path.
- Construction stays on the CPU. There are no deletions, filters, cosine or
  inner-product modes, dynamic shapes, or general large-index paging.
- The carrier and patch contract are specific to NPU3720 and graph compiler
  8.3.

The implementation follows the hierarchy described by
[Malkov and Yashunin](https://arxiv.org/abs/1603.09320), with a deliberately
simple bounded neighbor-selection policy suitable for this demonstration.
