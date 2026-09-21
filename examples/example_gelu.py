import os

import numpy as np
import npunlock as npu


gelu_c: bytes = b"""
/* Calibrated contiguous/static FP16 ACT entry, derived from marker-dims.c.
 * Metadata validation/general stride/layout handling are not established.
 * Reads the ACT input (already DPU add(x,bias)), not the graph's host input.
 */
static __attribute__((always_inline)) inline unsigned load32(const unsigned char *p) {
    return (unsigned)p[0] | ((unsigned)p[1] << 8) |
           ((unsigned)p[2] << 16) | ((unsigned)p[3] << 24);
}

void controlled_act(unsigned layerParams) {
    const unsigned char *params = (const unsigned char *)layerParams;
    unsigned rank = load32(params + 0xc);
    const unsigned char *dims = (const unsigned char *)load32(params + 0x10);
    if (rank == 0 || rank > 15 || !dims) return;
    unsigned count = 1;
    for (unsigned d = 0; d < rank; ++d) {
        unsigned dim = load32(dims + d * 4);
        if (dim == 0) return;
        count *= dim;
    }
    const __fp16 *in = (const __fp16 *)load32(params);
    __fp16 *out = (__fp16 *)load32(params + 0x28);
    const float SQRT_2_DIV_PI = 0.7978845608028654f;
    for (unsigned i = 0; i < count; ++i) {
      float x = (float) in[i];
      float w = x + 0.044715f * x * x * x;
      w = w * SQRT_2_DIV_PI;
      w = tanhf(w);
      out[i] = (__fp16) (0.5f * x * (1.0f + w));
    }
}
void strtof() {}
void __truncdfsf2() {}
typedef unsigned long uint32_t;
long long __fixsfdi(float x) {}
"""


def gelu_reference(value: np.ndarray) -> np.ndarray:
    value_f32 = value.astype(np.float32)
    return (
        0.5
        * value_f32
        * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (value_f32 + 0.044715 * value_f32**3)))
    )


def main() -> None:

    native_dir = os.environ.get(
        "NPUNLOCK_NATIVE_DIR", R"D:\srcs\npunlock\build\windows\Debug"
    )
    if not native_dir:
        raise RuntimeError(
            "set NPUNLOCK_NATIVE_DIR to the directory containing the built "
            "npunlock DLLs"
        )

    # npu.compile() reads NPUNLOCK_MOVITOOLS_DIR automatically. Alternatively,
    # configure the same directory in Python before compiling:
    npu.configure(movi_dll_dir=R"D:\Drivers\NPU\MVC_DEPEND\bin")
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
    program = npu.compile(
        npu.Graph(inputs=[x], outputs=[y], name="gelu_example"),
        native_dir=native_dir,
    )

    input_value = np.linspace(-4, 4, N, dtype=np.float16).reshape(1, -1)
    outputs = program.run({"x": input_value})
    reference = gelu_reference(input_value).astype(np.float16)
    max_abs_error = np.max(np.abs(outputs["y"] - reference))
    print(f"maximum absolute error: {max_abs_error:g}")


if __name__ == "__main__":
    main()
