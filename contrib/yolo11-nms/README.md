# YOLO11n NMS on the NPU

This example runs the official YOLO11n detector on an Intel Meteor Lake NPU
and replaces the usual CPU non-maximum suppression (NMS) step with a custom
ACT-SHAVE kernel built by npunlock.

Two npunlock paths are included:

- a straightforward two-workload version, where YOLO and candidate selection
  run first and the custom NMS graph runs second;
- an experimental single-graph version, where YOLO, selection, packing, and
  NMS complete in one NPU submission.

The single-graph path is deliberately tied to the reviewed YOLO11n model and
the recorded NPU compiler/driver stack. It fails closed when those hashes or
contracts do not match.

## Try it

From the repository root on the supported Windows/NPU3720 setup:

```powershell
python -m venv build\yolo11-nms-venv
$python = '.\build\yolo11-nms-venv\Scripts\python.exe'
& $python -m pip install -r contrib\yolo11-nms\requirements.txt
& $python -m pip install .

& $python contrib\yolo11-nms\scripts\export_model.py `
  --output-dir contrib\yolo11-nms\artifacts\model
& $python contrib\yolo11-nms\scripts\download_sample.py `
  --output-dir contrib\yolo11-nms\artifacts\images

& $python contrib\yolo11-nms\scripts\run_demo.py `
  --model-dir contrib\yolo11-nms\artifacts\model `
  --image contrib\yolo11-nms\artifacts\images\bus.jpg `
  --output-dir contrib\yolo11-nms\outputs `
  --movi-dll-dir C:\path\to\MVC_DEPEND `
  --warmup 5 `
  --runs 20 `
  --require-single-graph
```

Add `--native-dir build\windows\Debug` when using an uninstalled development
build. The demo writes a JSON report and annotated images under `outputs/`.
Model weights, ONNX exports, native graphs, images, and reports are generated
locally and are not committed.

## What was verified

On the recorded Core Ultra 5 125H system, OpenVINO 2026.3.1, Intel NPU driver
32.0.100.5540, and NPU graph compiler 8.3:

- the `bus.jpg` run found the same bus and four people as the CPU reference;
- the `zidane.jpg` run found the same two people and one tie;
- source candidate indices and class order matched exactly;
- maximum box drift was 0.25 pixels in 640x640 network coordinates, with zero
  confidence drift; and
- the complete hardware-gated suite passed 15 tests.

Median end-to-end time over the retained 50-run measurement was 16.10 ms for
the OpenVINO-plus-CPU-NMS baseline, 21.91 ms for the two-workload npunlock
path, and 19.42 ms for the experimental single graph. These are functional
results, not a speedup claim.

## Public smoke test

The small public suite checks the NMS behavior and transport layout without an
NPU or MoviTools:

```powershell
$env:PYTHONPATH = (Resolve-Path contrib\yolo11-nms)
& $python -m unittest contrib\yolo11-nms\tests\test_reference.py -v
```

## Limits

- Windows x64, Meteor Lake/NPU3720, batch 1, and static 640x640 input.
- At most 255 pre-NMS candidates and 255 returned detections.
- Candidate transport is FP16.
- The two-workload path crosses one host-visible candidate boundary.
- The single-graph binding supports only the exact reviewed model and software
  stack; it never guesses ACT positions for a different graph.
- Preprocessing and visualization remain on the CPU.

Ultralytics code and weights keep their upstream licensing terms. This
repository does not redistribute the checkpoint or generated models.
