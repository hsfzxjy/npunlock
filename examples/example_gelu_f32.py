import numpy as np
import npunlock as npu


gelu_c: bytes = b"""
#define MLIBM_DEFINE_LINK_COMPAT 1
#include <npunlock/npu3720_kernel.h>

void controlled_act(unsigned layerParams) {
    act_abi_invocation invocation;
    ACT_ABI_LOAD_INVOCATION32_OR_RETURN(layerParams, invocation);
    const float *in = ACT_ABI_INPUT_PTR32(const float, invocation, 0u);
    float *out = ACT_ABI_OUTPUT_PTR32(float, invocation, 1u);
    const float SQRT_2_DIV_PI = 0.7978845608028654f;
    for (unsigned i = 0; i < invocation.element_count; ++i) {
      float x = in[i];
      float w = x + 0.044715f * x * x * x;
      w = w * SQRT_2_DIV_PI;
      w = tanhf(w);
      out[i] = 0.5f * x * (1.0f + w);
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
    x = npu.input("x", shape=(1, N), dtype="f32")
    y = npu.custom(
        x,
        source=gelu_c,
        carrier="Abs",
        _name="y",
    )
    program = npu.compile(npu.Graph(inputs=[x], outputs=[y], name="gelu_f32_example"))

    input_value = np.linspace(-4, 4, N, dtype=np.float32).reshape(1, -1)
    outputs = program.run({"x": input_value})
    reference = gelu_reference(input_value).astype(np.float32)
    max_abs_error = np.max(np.abs(outputs["y"] - reference))
    print(f"maximum absolute error: {max_abs_error:g}")


if __name__ == "__main__":
    main()
