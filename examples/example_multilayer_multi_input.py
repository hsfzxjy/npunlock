import numpy as np
import npunlock as npu


weighted_mix_c: bytes = b"""
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
static __attribute__((always_inline)) inline unsigned load32(const unsigned char *p) {
    return (unsigned)p[0] | ((unsigned)p[1] << 8) |
           ((unsigned)p[2] << 16) | ((unsigned)p[3] << 24);
}

void controlled_act(unsigned layerParams) {
    const unsigned char *params = (const unsigned char *)layerParams;
    unsigned rank = load32(params + 0x0c);
    const unsigned char *dims = (const unsigned char *)load32(params + 0x10);
    if (rank == 0 || rank > 15 || !dims) return;

    unsigned count = 1;
    for (unsigned d = 0; d < rank; ++d) {
        unsigned dim = load32(dims + d * 4);
        if (dim == 0 || dim > 32u / count) return;
        count *= dim;
    }

    const __fp16 *a = (const __fp16 *)load32(params + 0x00);
    const __fp16 *b = (const __fp16 *)load32(params + 0x28);
    __fp16 *out = (__fp16 *)load32(params + 0x50);
    for (unsigned i = 0; i < count; ++i) {
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
    # npu.configure(movi_dll_dir=r"D:\path\containing\MoviTools\DLLs")
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
