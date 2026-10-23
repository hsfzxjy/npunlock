# YOLO11n with ACT-SHAVE NMS

This contribution runs the official pretrained Ultralytics YOLO11n detection
network on an Intel Meteor Lake NPU and runs confidence filtering, deterministic
ranking, class-aware IoU suppression, and output limiting on the NPU through an
npunlock ACT-SHAVE kernel.

The checkpoint is obtained through the documented
[Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/) interface, and
the two ONNX variants use the official
[export API](https://docs.ultralytics.com/modes/export/). Ultralytics code and
weights retain their upstream AGPL-3.0 or Enterprise licensing terms; this
repository does not redistribute either the checkpoint or generated models.

It compares four configurations with the same 640x640 input, confidence
threshold, IoU threshold, pre-NMS capacity, and maximum detection count:

| Configuration | Network | Candidate selection / NMS |
| --- | --- | --- |
| A. OpenVINO baseline | OpenVINO `NPU` | CPU deterministic reference |
| B. Ultralytics embedded NMS | OpenVINO `NPU` attempt | Exported ONNX NMS subgraph |
| C. OpenVINO + npunlock | OpenVINO `NPU`, with static preselection in the same graph | npunlock ACT-SHAVE NMS |
| D. Experimental single graph | OpenVINO-format YOLO/static-preselection graph compiled through the NPU UMD | npunlock ACT-SHAVE NMS patched into that native graph |

Configuration C uses **two separate NPU workloads**. The YOLO network and
static preselector are one OpenVINO graph, but the npunlock NMS carrier remains
a second graph:

```text
image --CPU letterbox/normalize--> YOLO11n + static preselector OpenVINO NPU graph
      --host-visible 255-candidate fixed tensor / pack / duplicate-->
      npunlock ACT-SHAVE NMS graph
      --host unpack/visualize--> detections
```

The raw `(1,84,8400)` YOLO tensor stays inside the first NPU graph.

Configuration D is a genuine **single native NPU graph**:

```text
image + four-word threshold header --> YOLO11n --> TopK/pack --> patched ACT NMS
                                   one submission / one fence
                                   --> fixed detection tensor
```

OpenVINO reads and constructs the static graph and serializes IR. npunlock's
`ir2blob` C API asks the NPU UMD to compile that IR, `patchblob` substitutes
the reviewed carrier ranges, and `graphinfer` executes the result. D is not an
OpenVINO Runtime infer request, and it is not two NPU graphs presented as one.
No intermediate detection tensor crosses the host boundary.

This path is deliberately narrower than C. Explicit ACT positions are accepted
only if the model hash, static weights, NPU driver/compiler provenance, and
complete unpatched native-graph hash match the reviewed manifest under
`bindings/`. Any mismatch fails closed. General discovery currently rejects an
unrelated ACT group in the large YOLO graph because that group has inconsistent
observed tensor contracts; this contribution does not weaken that safety check
or claim general arbitrary-graph injection.

The host performs preprocessing, orchestration, memory transfers, output
unpacking, and visualization. It also packs C's intermediate candidate tensor.
In D, record packing is part of the NPU graph and the host supplies only the
four-word header. The host makes no confidence, ranking, IoU, or suppression
decisions in C or D.

## Why this is interesting

Ultralytics can export YOLO11n with `nms=True`, and that exported model is valid:
it executes on the CPU and returns `(1,100,6)` for the recorded `max_det=100`
export. On the hardware and software versions recorded below, explicit NPU
compilation fails inside the NPU compiler:

```text
IE.Reshape doesn't support dynamic shapes
```

No `AUTO`, `HETERO`, or CPU fallback device is used. The failure is specific to
the tested OpenVINO/NPU compiler/driver combination. It does not show that NMS,
embedded-NMS exports, or other Intel NPUs are generally unsupported.

The raw YOLO11n graph and the raw graph extended with static preselection both
compile and execute directly on `NPU`. npunlock then provides a programmable
bounded NMS implementation even though the complete exported NMS graph is
rejected by this compiler version.

## Fixed-capacity design

The raw YOLO11n output is `(1,84,8400)`: four `xywh` coordinates and 80 COCO
class scores per candidate. A static OpenVINO subgraph attached directly to the
YOLO output and compiled in the same explicit NPU workload performs:

1. per-candidate class score maximum and class ID selection;
2. stable global TopK over all 8,400 candidates;
3. gathering of boxes, class IDs, scores, and original candidate indices; and
4. `xywh` to `xyxy` conversion.

The current ACT contract exposes 2,048 FP16 values (4 KiB) to each invocation.
The NMS input therefore contains an eight-word header plus 255 eight-word
candidate records. C duplicates the 2,048-word buffer into a 4,096-word unary
carrier. D duplicates it four times into an 8,192-word carrier; on the reviewed
full-model compilation this produces four 2,048-word carrier invocations.
Every patched invocation therefore receives one complete candidate set within
its advertised span. The kernel never reads across an invocation boundary.

Each record is:

```text
x1, y1, x2, y2, confidence, class_id, source_index_low8, source_index_high8
```

`kernels/nms.c` applies the confidence threshold, repeatedly selects the
highest remaining confidence, uses the lower original candidate index as a
deterministic tie-breaker, suppresses same-class boxes when `IoU > threshold`,
and stops at `max_det`. Results use the same fixed-capacity representation.

Configurations A, C, and D deliberately use the same `pre_nms_topk=255`
contract. This is not identical to Ultralytics' much larger default pre-NMS
cap in pathological dense scenes. It is a documented consequence of the
currently validated ACT-local span, not a general YOLO or hardware limit.

## Requirements

- Windows x64
- Meteor Lake / Intel NPU3720
- an installed Intel NPU driver
- Python 3.12 (the recorded run used 3.12.10)
- the repository's normal MSVC/CMake build requirements
- caller-supplied MoviTools `MVC_DEPEND` root

The Python dependencies are pinned in `requirements.txt`. Model weights,
generated ONNX files, native blobs, images, and reports are ignored and are not
committed.

From the repository root:

```powershell
python -m venv build\yolo11-nms-venv
.\build\yolo11-nms-venv\Scripts\python.exe -m pip install -r contrib\yolo11-nms\requirements.txt
.\build\yolo11-nms-venv\Scripts\python.exe -m pip install .
```

Do not install or downgrade an OEM driver to obtain MoviTools. Supply an
already-extracted `MVC_DEPEND` root as required by npunlock.

## Export and run

The export script downloads the official `yolo11n.pt` checkpoint through
Ultralytics and creates static raw-output and embedded-NMS ONNX files:

```powershell
$python = '.\build\yolo11-nms-venv\Scripts\python.exe'
& $python contrib\yolo11-nms\scripts\export_model.py `
  --output-dir contrib\yolo11-nms\artifacts\model

& $python contrib\yolo11-nms\scripts\download_sample.py `
  --output-dir contrib\yolo11-nms\artifacts\images
```

Run all four configurations and 20 measured iterations after five warm-ups:

```powershell
& $python contrib\yolo11-nms\scripts\run_demo.py `
  --model-dir contrib\yolo11-nms\artifacts\model `
  --image contrib\yolo11-nms\artifacts\images\bus.jpg `
  --output-dir contrib\yolo11-nms\outputs `
  --movi-dll-dir C:\path\to\MVC_DEPEND `
  --warmup 5 `
  --runs 20 `
  --require-single-graph
```

For a development build instead of an installed package, add the directory
containing `npunlock.dll` and `npunlock_worker.exe`:

```powershell
  --native-dir build\windows\Debug
```

Outputs are:

- `outputs/report.json`: machine-readable versions, hashes, device placement,
  compatibility result, detections, patch evidence, and timings;
- `outputs/detections_npunlock.jpg`: visualized configuration-C detections;
- `outputs/detections_single_graph.jpg`: visualized configuration-D detections;
- `outputs/single-graph-build/`: generated IR used for the hash-bound D build.

Omit `--require-single-graph` to retain A-C results when the installed stack
does not match the reviewed D binding. Use `--skip-single-graph` to skip D.
Neither mode silently falls back to CPU.

## Tests

Offline reference and layout tests need no NPU or MoviTools:

```powershell
$env:PYTHONPATH = (Resolve-Path contrib\yolo11-nms)
& $python -m unittest discover -s contrib\yolo11-nms\tests -p 'test_reference.py' -v
```

Hardware tests are explicitly gated:

```powershell
$env:NPUNLOCK_ENABLE_NPU_TESTS = '1'
$env:NPUNLOCK_MOVITOOLS_DIR = 'C:\path\to\MVC_DEPEND'
$env:YOLO11_MODEL_DIR = (Resolve-Path contrib\yolo11-nms\artifacts\model)
$env:YOLO11_SAMPLE_DIR = (Resolve-Path contrib\yolo11-nms\artifacts\images)
$env:PYTHONPATH = (Resolve-Path contrib\yolo11-nms)
& $python -m unittest discover -s contrib\yolo11-nms\tests -p 'test_*.py' -v
```

`test_npu_operator.py` covers empty input, one detection, non-overlap,
same-class suppression, different-class preservation, equal confidences,
threshold boundaries, and the full 255-candidate capacity. The end-to-end test
uses real YOLO11n predictions and requires exact source-index/class agreement,
at most 0.5 network-pixel box drift, and at most 0.002 confidence drift. It
checks both the established two-workload C path and the one-workload D path.

## Recorded result

Recorded on 2026-10-23:

| Component | Version |
| --- | --- |
| CPU | Intel Core Ultra 5 125H |
| NPU | Intel AI Boost / device `0x7D1D` |
| Intel NPU driver | `32.0.100.5540` (`1005540` reported by OpenVINO) |
| OpenVINO | `2026.3.1-22476-759c5a6ab8c-releases/2026/3` |
| Ultralytics | `8.4.183` |
| NPU graph compiler | `8.3` |
| npunlock base commits | `7da7365`, `f4e088a`, plus this single-graph follow-up |

Model and sample hashes:

| Artifact | SHA-256 |
| --- | --- |
| `yolo11n.pt` | `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1` |
| raw ONNX | `7e69e46112e98b712b77df004708c5a0fbf1bfcf4021176a2ec2cc44231b65b4` |
| embedded-NMS ONNX | `63d55d3bd2e7b9a20ddf185e7454fecbf105be36744b4de2457c907b7eabaf6b` |
| Ultralytics `bus.jpg` | `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63` |
| Ultralytics `zidane.jpg` | `16d73869e3267a7d4ed00de8e860833bd1657c1b252e94c0c348277adc7b6edb` |

With `conf=0.25`, `iou=0.7`, `max_det=100`, ten warm-ups, and 50 runs:

| Measurement | Median |
| --- | ---: |
| A: OpenVINO NPU network | 9.06 ms |
| A: raw-output host copy | 0.739 ms |
| A: CPU candidate preselection | 1.15 ms |
| A: CPU NMS | 0.214 ms |
| A: complete end-to-end wall time | 16.10 ms |
| C: fused OpenVINO NPU network + preselector | 10.27 ms |
| C: selected-candidate host copy | 0.0597 ms |
| C: npunlock NMS execution/fence wall time | 2.28 ms |
| C: host candidate pack | 0.112 ms |
| C: NMS input/output shared-buffer copies | 0.0035 / 0.0092 ms |
| C: complete end-to-end wall time | 21.91 ms |
| D: single native graph execution/fence wall time | 11.42 ms |
| D: image + header shared-buffer copies | 0.367 ms |
| D: detection-buffer copy | 0.0160 ms |
| D: complete end-to-end wall time | 19.42 ms |

The npunlock execution API does not expose device-only ACT profiling. The C
and D execution measurements are submission-to-fence wall times using shared
Level Zero buffers, excluding separately measured host copies. They must not
be described as pure kernel time. D's retained complete-pipeline median was
lower than C's, but a repeated run during development varied as high as 28.12
ms because host preprocessing/copy scheduling varied. No speedup claim is made.

On `bus.jpg`, configurations A, C, and D produced the same five source candidate
indices and class order (one bus, four people); maximum box difference was
0.25 pixels in the 640x640 network coordinate space and confidence difference
was zero. On `zidane.jpg`, all three produced the same three source
indices/classes (two people and one tie), with 0.125-pixel maximum box
difference and zero confidence difference. The kernel also matched the
independent FP16 reference exactly in the operator tests. The complete opt-in
suite passed 15/15 tests.

The retained npunlock patch reports record a validated linked SHAVE ELF hash,
four changed ACT range/code targets, graph compiler 8.3/NPU3720 provenance,
and successful section-content and mutation-set preservation checks. This,
together with exclusive `NPU` device reports and the host-oracle comparison,
is the execution evidence; graph creation alone is not treated as correctness.
For D, the report identifies unpatched native graph
`f0e4710edae2777f72807bd06369c202bcfe1644684fa8f2cb4ac1e655ed16f3`
and invocations/ranges 441-444, each validated as one FP16 input and one FP16
output of 2,048 elements. The single submission then produced the same
real-image detections as the independent A pipeline.

## Current limitations

- Only batch 1, static 640x640 YOLO11 detection output is demonstrated.
- Pre-NMS capacity is 255 candidates and `max_det` cannot exceed 255.
- Candidate transport into the ACT kernel is FP16. Boxes and scores are
  compared against an explicitly FP16-quantized operator reference.
- C remains two NPU workloads. Its 255 selected candidates cross a host-visible
  boundary; the raw 8,400-candidate tensor does not.
- D is one workload, but currently only for the exact reviewed model,
  OpenVINO, driver, graph compiler, and native-graph hash. New stacks require
  a new independently reviewed binding; positions are never inferred from a
  source node name or from merely being the last ACT ranges.
- The recorded driver could not keep the separate C NMS session and the full D
  session initialized concurrently. The runner closes C before creating D.
- Kernel-only hardware timing is unavailable through the current graphinfer
  API.
- Results apply only to the recorded Meteor Lake driver/compiler combination.
- Model preprocessing and visualization remain on the CPU.

These constraints are intentional. Extending the capacity requires another
validated global/merge contract rather than reading beyond an ACT invocation's
advertised span.
