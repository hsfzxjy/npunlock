import numpy as np
import npunlock as npu


gelu_c: bytes = b"""
#include <npunlock/npu3720_kernel.h>

/* Calibrated contiguous/static FP16 ACT entry, derived from marker-dims.c.
 * Metadata validation/general stride/layout handling are not established.
 * Reads the ACT input (already DPU add(x,bias)), not the graph's host input.
 */
NPUNLOCK_NPU3720_MLIBM_DEFINE_LINK_COMPAT()

void controlled_act(unsigned layerParams) {
    npunlock_npu3720_act_abi_invocation invocation;
    NPUNLOCK_NPU3720_ACT_ABI_LOAD_INVOCATION32_OR_RETURN(layerParams, 2048u, invocation);
    const __fp16 *in =
        NPUNLOCK_NPU3720_ACT_ABI_INPUT_PTR32(const __fp16, invocation, 0u);
    __fp16 *out = NPUNLOCK_NPU3720_ACT_ABI_OUTPUT_PTR32(__fp16, invocation, 1u);
    const float SQRT_2_DIV_PI = 0.7978845608028654f;
    for (unsigned i = 0; i < invocation.element_count; ++i) {
      float x = (float) in[i];
      float w = x + 0.044715f * x * x * x;
      w = w * SQRT_2_DIV_PI;
      w = tanhf(w);
      out[i] = (__fp16) (0.5f * x * (1.0f + w));
    }
}
"""


def gelu_reference(value: np.ndarray) -> np.ndarray:
    value_f32 = value.astype(np.float32)
    return (
        0.5
        * value_f32
        * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (value_f32 + 0.044715 * value_f32**3)))
    )


def main() -> None:
    # npu.compile() reads NPUNLOCK_MOVITOOLS_DIR automatically. Alternatively,
    # configure the same directory in Python before compiling:
    # npu.configure(movi_dll_dir=r"C:\path\to\MVC_DEPEND")
    N = 2048
    x = npu.input("x", shape=(1, N), dtype="f16")
    y = npu.custom(
        x,
        source=gelu_c,
        carrier="Abs",
        _shape=x.shape,
        _dtype=x.dtype,
        _name="y",
    )
    program = npu.compile(npu.Graph(inputs=[x], outputs=[y], name="gelu_example"))

    input_value = np.linspace(-4, 4, N, dtype=np.float16).reshape(1, -1)
    outputs = program.run({"x": input_value})
    reference = gelu_reference(input_value).astype(np.float16)
    max_abs_error = np.max(np.abs(outputs["y"] - reference))
    print(f"maximum absolute error: {max_abs_error:g}")


if __name__ == "__main__":
    main()
