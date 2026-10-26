# ByteTrack assignment on ACT-SHAVE

This example tracks people in real video with ByteTrack while a custom
ACT-SHAVE kernel solves every non-empty ByteTrack linear-assignment stage on
Intel NPU3720. Empty stages are completed directly from their shape because
there is no optimization problem or device work to perform. For non-empty
stages, the kernel performs the complete bounded shortest-augmenting-
path search: it initializes assignments, updates dual costs, follows
predecessors, augments the matching, and writes both row and column results.

That computation is a useful contrast with ordinary inference operators. The
tested OpenVINO stack has no dedicated linear-assignment operation, and
ByteTrack's dynamically changing augmenting paths do not have a practical
stock-operator formulation in this example. This is not a claim that a bounded
solver is theoretically impossible to unroll into some larger static graph.

## What runs where

The demo deliberately describes the split as it is:

```text
video decode, letterbox, annotation                         CPU
YOLO11n and stable top-k candidate selection                OpenVINO NPU
class filtering, greedy NMS, IoU costs, Kalman prediction   CPU
non-empty ByteTrack linear-assignment stages                ACT-SHAVE NPU
empty assignment contracts                                  CPU
track activation, reactivation, loss and removal            CPU
```

The assignment input is a runtime-supplied FP32 matrix, not code generated for
one frame. The current public contract supports rectangular matrices through
16 by 16. An explicit bit mask marks forbidden pairs; NaN, infinity, and costs
above the selected threshold are also forbidden. Unmatched rows and columns
use the same `threshold / 2` square augmentation as `lap.lapjv(...,
extend_cost=True, cost_limit=threshold)`, so this is not unconstrained matching
followed by threshold filtering.

The graph uses the validated two-input FP32 `Subtract` carrier. Four ACT
invocations receive independent copies of the same 288-word problem and must
return identical result bytes. The second carrier input is a validated zero
guard; the solver reads costs from the first input. Replica disagreement,
malformed output, duplicate assignments, forbidden matches, and inconsistent
unmatched sets all fail closed on the host.

## Run the video demo

From the repository root on the supported Windows/NPU3720 system:

```powershell
python -m venv build\bytetrack-venv
$python = '.\build\bytetrack-venv\Scripts\python.exe'
& $python -m pip install -r contrib\bytetrack\requirements.txt
& $python -m pip install .

& $python contrib\yolo11-nms\scripts\export_model.py `
  --output-dir contrib\bytetrack\artifacts\model
& $python contrib\bytetrack\scripts\download_sample.py `
  --output contrib\bytetrack\artifacts\vtest.avi

& $python contrib\bytetrack\scripts\run_demo.py `
  --video contrib\bytetrack\artifacts\vtest.avi `
  --model-dir contrib\bytetrack\artifacts\model `
  --output-dir contrib\bytetrack\outputs `
  --movi-dll-dir C:\path\to\MVC_DEPEND `
  --native-dir build\windows\Debug `
  --max-frames 300
```

The model exporter downloads the official YOLO11n checkpoint under its
upstream terms. The sample downloader fetches OpenCV's `vtest.avi` pedestrian
clip. Neither model weights nor video data are committed. The runner writes an
annotated MP4 and a JSON report with assignment patch evidence, persistent
track counts, CPU-reference agreement, and component timings.

By default the demo tracks COCO class 0 (person). Repeat `--class-id` to select
other classes. `--skip-cpu-reference` disables the parallel LAPJV tracker only
after the deployment has completed its own acceptance checks.

## Correctness tests

The public suite needs no NPU, MoviTools, model, or video:

```powershell
$env:PYTHONPATH = (Resolve-Path contrib\bytetrack)
& $python -m unittest discover `
  -s contrib\bytetrack\tests -p 'test_*.py' -v
```

It covers empty, scalar, rectangular, tied, forbidden, non-finite, random,
adversarial, and maximum-capacity assignment problems. Small cases are checked
by enumeration; SciPy solves the augmented reference problem; and LAPJV is an
additional independent baseline. Tracking tests cover all three association
stages, low-confidence recovery, unconfirmed tracks, reactivation, removal,
and identity persistence.

## Benchmark

```powershell
& $python contrib\bytetrack\scripts\benchmark.py `
  --movi-dll-dir C:\path\to\MVC_DEPEND `
  --native-dir build\windows\Debug `
  --output contrib\bytetrack\outputs\benchmark.json `
  --warmup 30 --runs 500 --cpu-runs 5000
```

On the recorded Core Ultra 5 125H system with graph compiler 8.3, OpenVINO
2026.3.1, and the repository's current native runtime, median complete NPU
calls were 0.57 ms for 1 by 1, 0.58 ms for 8 by 8, and 0.80 ms for 16 by 16.
The benchmark also includes empty contracts plus 1 by 16, 4 by 16, 8 by 16,
16 by 4, and 16 by 8 problems. Median SciPy and LAPJV calls were about
0.10-0.12 ms. The NPU result
demonstrates the control-flow capability and reduces CPU algorithm work; it is
not a latency speedup over optimized CPU solvers at this capacity.

The optimized 300-frame `vtest.avi` run produced CPU/NPU tracker agreement on
every frame. Of 900 association stages, 549 had an empty side and completed
from their shape without submitting a graph; all 351 non-empty stages executed
on the NPU. YOLO inference took 9.58 ms median, the complete tracker took 1.87
ms, and non-empty assignment graph calls took 0.94 ms median from submission
through output copy. Complete decode-to-annotated-frame work took 19.05 ms
median. Persistent identities included tracks visible for 289 and 282 frames.
The tracker also uses an SPD Cholesky solve for its 4 by 4 Kalman updates.
These are one-system measurements, not general performance claims.

## Limits

- Windows x64, Meteor Lake/NPU3720, graph compiler 8.3, and static FP32 carrier
  tensors.
- At most 16 rows and 16 columns in each high-confidence, low-confidence, or
  unconfirmed association stage. The video runner caps post-NMS detections at
  16; the tracker rejects overflow instead of truncating internally.
- Association is class-agnostic, matching the selected original ByteTrack
  reference. Filtering before the tracker controls which classes share it.
- YOLO preprocessing, NMS, Kalman filtering, cost construction, empty
  assignment contracts, lifecycle management, video I/O, and drawing remain
  on the CPU.
- The ordinary execution path copies through graphinfer's internal Level Zero
  allocations. Kernel-only device time is not exposed by this demo.
- `vtest.avi` has no MOT ground truth, so the recorded run reports exact
  CPU/NPU state agreement and identity longevity, not MOTA, IDF1, or HOTA.
- A successful fence is never accepted as semantic proof: device results are
  structurally validated, replica-compared, and checked against LAPJV in the
  default demo configuration.

The tracker behavior follows the original
[FoundationVision ByteTrack](https://github.com/FoundationVision/ByteTrack)
pipeline. The CPU baseline uses the BSD-2-Clause
[`lap`](https://github.com/gatagat/lap) package. This repository does not copy
or redistribute either project's model weights or datasets.
