# XGBoost inference on ACT-SHAVE

This example runs a numerical XGBoost binary classifier inside a custom
ACT-SHAVE kernel on Intel Meteor Lake. Tree traversal, missing-value routing,
leaf accumulation, sigmoid, and class selection happen on the NPU.

The demo uses the Wisconsin breast-cancer dataset: 569 rows, 30 features, and
a deterministic train/test split. It supports two useful styles:

- generate C specialized for one forest;
- keep a generic kernel and upload the forest through a second NPU input.

`XGBoostRuntime` selects the validated single-stage or connected two-part
runtime automatically, balances each graph chunk across four independent
invocation-local spans, and splits larger host batches while preserving order.

```python
from xgboost_npu import XGBoostRuntime

with XGBoostRuntime(
    "model.json",
    movi_dll_dir=r"C:\path\to\MVC_DEPEND",
    native_dir=r"build\windows\Debug",
    cache_dir=r"build\xgboost-program-cache",
) as runtime:
    probabilities = runtime.predict_proba(features)
    labels = runtime.predict(features)
```

## Try it

```powershell
python -m venv build\xgboost-venv
$python = '.\build\xgboost-venv\Scripts\python.exe'
& $python -m pip install -r contrib\xgboost\requirements.txt
& $python -m pip install .

& $python contrib\xgboost\scripts\train_model.py `
  --output-dir contrib\xgboost\artifacts\model

& $python contrib\xgboost\scripts\run_runtime_demo.py `
  --model-dir contrib\xgboost\artifacts\model `
  --output contrib\xgboost\outputs\runtime-report.json `
  --movi-dll-dir C:\path\to\MVC_DEPEND `
  --native-dir build\windows\Debug `
  --cache-dir build\xgboost-program-cache `
  --warmup 5 `
  --runs 20
```

The scripts keep trained models, generated kernels, native graphs, and reports
under ignored `artifacts/` and `outputs/` directories.

## What was verified

The recorded 32-tree, depth-3 model contains 372 nodes. It reached 97.20%
held-out accuracy and 0.9975 ROC AUC. Across all 143 held-out samples, the
specialized FP16, specialized FP32, and runtime-loaded FP32 NPU paths agreed
with official XGBoost on every class.

Maximum raw-margin error was 0.001945 for FP16 transport and 4.77e-7 for both
FP32 paths. The generic runtime kernel linked to a 3,472-byte SHAVE image; the
forest stayed in a separate ordinary NPU input.

On the recorded system, balancing a 32-row batch across four ACT invocations
reduced submission-through-fence wall time by 53% for the 502-node runtime
model and 52% for the connected 612-node model, with byte-identical results.

This is primarily a programmability example. The small recorded workload did
not establish a general performance win over the optimized CPU implementation.
See [benchmarks/README.md](benchmarks/README.md) for the runnable measurements.

## Public smoke test

The public suite checks JSON loading, XGBoost split/NaN semantics, generated C,
and the transport layout without using an NPU or MoviTools:

```powershell
$env:PYTHONPATH = "$(Resolve-Path python);$(Resolve-Path contrib\xgboost)"
& $python -m unittest discover -s contrib\xgboost\tests -p 'test_model.py' -v
```

## Supported model shape

- portable XGBoost JSON with `gbtree` and `binary:logistic`;
- numerical `<` splits and NaN default directions;
- one tree per boosting round and one output group;
- FP16 or FP32 feature/result transport;
- up to 509 nodes in one runtime forest, or 1,018 nodes in the validated
  connected two-part form;
- up to 64 trees in the connected form;
- 136 rows per single-stage graph execution or 128 rows per connected graph
  execution for the demonstrated 30-feature model.

Categorical splits, multiclass models, DART, random-forest layouts, custom
objectives, arbitrary model sizes, and arbitrary ACT layouts are not supported.
The normal OpenVINO ONNX route is included as a comparison, but OpenVINO
2026.3.1 could not import the exported `TreeEnsembleClassifier` model.
