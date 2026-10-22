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

It compares three configurations with the same 640x640 input, confidence
threshold, IoU threshold, pre-NMS capacity, and maximum detection count:

| Configuration | Network | Candidate selection / NMS |
| --- | --- | --- |
| A. OpenVINO baseline | OpenVINO `NPU` | CPU deterministic reference |
| B. Ultralytics embedded NMS | OpenVINO `NPU` attempt | Exported ONNX NMS subgraph |
| C. OpenVINO + npunlock | OpenVINO `NPU` | static OpenVINO NPU preselector plus npunlock ACT-SHAVE NMS |

Configuration C uses **three separate workloads**. It is not a fused graph:

```text
image --CPU letterbox/normalize--> YOLO11n OpenVINO NPU
      --host-visible (1,84,8400) tensor--> static TopK/class selection NPU graph
      --host pack/duplicate--> npunlock ACT-SHAVE NMS graph
      --host unpack/visualize--> detections
```

The host performs orchestration, fixed-layout packing, memory transfers, output
unpacking, and visualization. It does not make confidence, ranking, IoU, or
suppression decisions in configuration C.

## Why this is interesting

Ultralytics can export YOLO11n with `nms=True`, and that exported model is valid:
it executes on the CPU and returns `(1,300,6)`. On the hardware and software
versions recorded below, explicit NPU compilation fails inside the NPU compiler:

```text
IE.Reshape doesn't support dynamic shapes
```

No `AUTO`, `HETERO`, or CPU fallback device is used. The failure is specific to
the tested OpenVINO/NPU compiler/driver combination. It does not show that NMS,
embedded-NMS exports, or other Intel NPUs are generally unsupported.

The raw YOLO11n graph does compile and execute directly on `NPU`. npunlock then
provides a programmable bounded NMS implementation even though the complete
exported NMS graph is rejected by this compiler version.

## Fixed-capacity design

The raw YOLO11n output is `(1,84,8400)`: four `xywh` coordinates and 80 COCO
class scores per candidate. A static OpenVINO graph running explicitly on the
NPU performs:

1. per-candidate class score maximum and class ID selection;
2. stable global TopK over all 8,400 candidates;
3. gathering of boxes, class IDs, scores, and original candidate indices; and
4. `xywh` to `xyxy` conversion.

The current ACT contract exposes 2,048 FP16 values (4 KiB) to each invocation.
The NMS input therefore contains an eight-word header plus 255 eight-word
candidate records. The 2,048-word buffer is duplicated into a logical
4,096-word unary carrier so every patched invocation receives one complete
candidate set within its advertised span. The kernel never reads across an
invocation boundary.

Each record is:

```text
x1, y1, x2, y2, confidence, class_id, source_index_low8, source_index_high8
```

`kernels/nms.c` applies the confidence threshold, repeatedly selects the
highest remaining confidence, uses the lower original candidate index as a
deterministic tie-breaker, suppresses same-class boxes when `IoU > threshold`,
and stops at `max_det`. Results use the same fixed-capacity representation.

Both A and C deliberately use the same `pre_nms_topk=255` contract. This is not
identical to Ultralytics' much larger default pre-NMS cap in pathological dense
scenes. It is a documented consequence of the currently validated ACT-local
span, not a general YOLO or hardware limit.

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

Run all three configurations and 20 measured iterations after five warm-ups:

```powershell
& $python contrib\yolo11-nms\scripts\run_demo.py `
  --model-dir contrib\yolo11-nms\artifacts\model `
  --image contrib\yolo11-nms\artifacts\images\bus.jpg `
  --output-dir contrib\yolo11-nms\outputs `
  --movi-dll-dir C:\path\to\MVC_DEPEND `
  --warmup 5 `
  --runs 20
```

For a development build instead of an installed package, add the directory
containing `npunlock.dll` and `npunlock_worker.exe`:

```powershell
  --native-dir build\windows\Debug
```

Outputs are:

- `outputs/report.json`: machine-readable versions, hashes, device placement,
  compatibility result, detections, patch evidence, and timings;
- `outputs/detections_npunlock.jpg`: visualized configuration-C detections.

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
at most 0.5 network-pixel box drift, and at most 0.002 confidence drift.

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
| npunlock commit | `88bd584` plus this contribution |

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
| A: OpenVINO NPU network | 8.93 ms |
| A: CPU candidate preselection | 1.12 ms |
| A: CPU NMS | 0.217 ms |
| A: complete end-to-end wall time | 18.86 ms |
| C: OpenVINO NPU preselector | 25.81 ms |
| C: npunlock shared execution/fence wall time | 1.85 ms |
| C: host pack | 0.109 ms |
| C: shared-buffer upload copy | 0.0047 ms |
| C: shared-buffer download copy | 0.0111 ms |
| C: complete end-to-end wall time | 46.35 ms |

The npunlock execution API does not expose device-only ACT profiling. The
reported 1.85 ms is submission-to-fence wall time using shared Level Zero
buffers, excluding the separately measured host upload/download copies. It
must not be described as pure kernel time.

On `bus.jpg`, configurations A and C produced the same five source candidate
indices and class order (one bus, four people); maximum box difference was
0.25 pixels in the 640x640 network coordinate space and confidence difference
was zero. On `zidane.jpg`, both produced the same three source indices/classes
(two people and one tie), with 0.125-pixel maximum box difference and zero
confidence difference. The kernel also matched the independent FP16 reference
exactly in the operator tests.

The retained npunlock patch report records a validated linked SHAVE ELF hash,
four changed ACT range/code targets, graph compiler 8.3/NPU3720 provenance,
and successful section-content and mutation-set preservation checks. This,
together with exclusive `NPU` device reports and the host-oracle comparison,
is the execution evidence; graph creation alone is not treated as correctness.

## Current limitations

- Only batch 1, static 640x640 YOLO11 detection output is demonstrated.
- Pre-NMS capacity is 255 candidates and `max_det` cannot exceed 255.
- Candidate transport into the ACT kernel is FP16. Boxes and scores are
  compared against an explicitly FP16-quantized operator reference.
- YOLO inference, static NPU preselection, and custom NMS are separate NPU
  workloads with host-visible intermediate buffers.
- The current preselector is correctness-oriented and substantially slower
  than CPU postprocessing on this system.
- Kernel-only hardware timing is unavailable through the current graphinfer
  API.
- Results apply only to the recorded Meteor Lake driver/compiler combination.
- Model preprocessing and visualization remain on the CPU.

These constraints are intentional. Extending the capacity requires another
validated global/merge contract rather than reading beyond an ACT invocation's
advertised span.
