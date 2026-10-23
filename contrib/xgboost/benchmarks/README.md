# Benchmarks

The contribution reports CPU latency, complete npunlock call time, and the NPU
submission-to-fence interval separately. The last value is wall time, not a
device-only kernel timestamp.

Run the main specialized-kernel comparison with:

```powershell
python contrib\xgboost\scripts\run_demo.py `
  --model-dir contrib\xgboost\artifacts\model `
  --output-dir contrib\xgboost\outputs `
  --movi-dll-dir C:\path\to\MVC_DEPEND `
  --native-dir build\windows\Debug
```

`benchmark_tree_counts.py` compares 1, 4, 8, 16, and 32-tree prefixes.
`run_runtime_demo.py` measures the generic runtime-loaded forest, and
`run_partitioned_runtime_demo.py` runs the connected two-part model. Each
script checks its predictions against official XGBoost before writing a report.

Generated JSON reports live under the ignored `outputs/` directory.
