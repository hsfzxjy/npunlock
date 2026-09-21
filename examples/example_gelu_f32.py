import numpy as np
import npunlock as npu


gelu_c: bytes = b"""
/* Calibrated contiguous/static FP32 ACT entry, derived from marker-dims.c.
 * Metadata validation/general stride/layout handling are not established.
 * The carrier is marked precision-sensitive and compiled in ACCURACY mode so
 * the UMD retains one FP32 ACT group instead of FP32->FP16->FP32 conversion.
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
    const float *in = (const float *)load32(params);
    float *out = (float *)load32(params + 0x28);
    const float SQRT_2_DIV_PI = 0.7978845608028654f;
    for (unsigned i = 0; i < count; ++i) {
      float x = in[i];
      float w = x + 0.044715f * x * x * x;
      w = w * SQRT_2_DIV_PI;
      w = tanhf(w);
      out[i] = 0.5f * x * (1.0f + w);
    }
}
void strtof() {}
void __truncdfsf2() {}
typedef unsigned long uint32_t;
typedef long int32_t;
typedef unsigned long long uint64_t;
long long __fixsfdi(float x) {
    union {
        float f;
        uint32_t u;
    } v = {x};

    uint32_t bits = v.u;
    uint32_t sign = bits >> 31;
    uint32_t frac = bits & 0x007fffff;
    int32_t exp = (int32_t)((bits >> 23) & 0xff) - 127;

    /* |x| < 1, including zero and subnormals. */
    if (exp < 0)
        return 0;

    /*
     * Restore the implicit leading 1:
     *
     *     1.frac * 2^exp
     *
     * mantissa therefore represents a fixed-point value with
     * 23 fractional bits.
     */
    uint64_t mantissa = (uint64_t)(frac | 0x00800000);

    uint64_t value;

    if (exp <= 23)
        value = mantissa >> (23 - exp);
    else
        value = mantissa << (exp - 23);

    if (sign)
        return -(long long)value;

    return (long long)value;
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
        _shape=x.shape,
        _dtype=x.dtype,
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
