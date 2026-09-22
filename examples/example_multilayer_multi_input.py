import numpy as np
import npunlock as npu


weighted_mix_c: bytes = b"""
#include <npunlock/npu3720_kernel.h>

/* Calibrated static dense FP16 binary ACT entry.
 *
 * For the validated two-input carrier, the tensor records are:
 *   input A: layerParams + 0x00
 *   input B: layerParams + 0x28
 *   output:  layerParams + 0x50
 *
 * This example is intentionally fixed to the (1, 32) graph below. The UMD
 * partitions its binary carrier into four invocations of eight elements.
 */
void controlled_act(unsigned layerParams) {
    act_abi_invocation invocation;
    ACT_ABI_LOAD_INVOCATION32_OR_RETURN(layerParams, 32u, invocation);
    const __fp16 *a = ACT_ABI_INPUT_PTR32(const __fp16, invocation, 0u);
    const __fp16 *b = ACT_ABI_INPUT_PTR32(const __fp16, invocation, 1u);
    __fp16 *out = ACT_ABI_OUTPUT_PTR32(__fp16, invocation, 2u);
    for (unsigned i = 0; i < invocation.element_count; ++i) {
        float lhs = (float)a[i];
        float rhs = (float)b[i];
        out[i] = (__fp16)(lhs * 0.75f + rhs * 0.25f);
    }
}
"""


def reference(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Mirror the graph's FP16 layer boundaries explicitly.
    abs_x = np.abs(x).astype(np.float16)
    abs_y = np.abs(y).astype(np.float16)
    mixed = (
        abs_x.astype(np.float32) * np.float32(0.75)
        + abs_y.astype(np.float32) * np.float32(0.25)
    ).astype(np.float16)
    return np.sqrt(mixed.astype(np.float32)).astype(np.float16)


def main() -> None:
    # npu.compile() reads NPUNLOCK_MOVITOOLS_DIR automatically. Alternatively:
    # npu.configure(movi_dll_dir=r"C:\path\to\MVC_DEPEND")
    shape = (1, 32)
    x = npu.input("x", shape=shape, dtype="f16")
    y = npu.input("y", shape=shape, dtype="f16")

    abs_x = npu.Abs(x, _shape=shape, _dtype="f16", _name="abs_x")
    abs_y = npu.Abs(y, _shape=shape, _dtype="f16", _name="abs_y")
    mixed = npu.custom(
        abs_x,
        abs_y,
        source=weighted_mix_c,
        carrier="Maximum",
        _shape=shape,
        _dtype="f16",
        _name="weighted_mix",
    )
    output = npu.Sqrt(mixed, _shape=shape, _dtype="f16", _name="output")

    graph = npu.Graph(
        inputs=[x, y],
        outputs=[output],
        name="multilayer_multi_input_example",
    )
    program = npu.compile(graph)

    x_value = np.linspace(-4.0, 3.5, 32, dtype=np.float16).reshape(shape)
    y_value = np.linspace(2.5, -3.0, 32, dtype=np.float16).reshape(shape)
    actual = program.run({"x": x_value, "y": y_value})["output"]
    expected = reference(x_value, y_value)
    max_abs_error = np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32)))
    print(f"maximum absolute error: {max_abs_error:g}")


if __name__ == "__main__":
    main()
