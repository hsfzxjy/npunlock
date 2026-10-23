# Kernel choices

The contribution offers two kernel styles.

The specialized path uses `scripts/generate_kernel.py` to turn one XGBoost
forest into readable C. Thresholds, leaves, and branches become part of the
compiled SHAVE image. Generated C is a build artifact and is not committed.

The runtime path keeps one model-independent kernel in
`xgboost_npu/runtime.py`. The complete compact forest arrives through a second
FP32 graph input, so a compatible model can be replaced without compiling a
new kernel. Larger supported forests use two connected traversal stages in one
graph.

Both paths are intentionally bounded and reject models outside the contract
described in the contribution's main [README](../README.md).
