import os

import numpy as np
import npunlock as npu


# TODO: Supply the inline ACT-SHAVE source as bytes. It must define the
# controlled_act entry point and implement the tanh GELU approximation used by
# gelu_reference() below for the invocation-local dense FP16 span.
gelu_c: bytes | None = None


def gelu_reference(value: np.ndarray) -> np.ndarray:
    value_f32 = value.astype(np.float32)
    return 0.5 * value_f32 * (
        1.0
        + np.tanh(
            np.sqrt(2.0 / np.pi)
            * (value_f32 + 0.044715 * value_f32**3)
        )
    )


def main() -> None:
    if gelu_c is None:
        raise NotImplementedError(
            "implement gelu_c as bytes containing the ACT-SHAVE C source"
        )

    native_dir = os.environ.get("NPUNLOCK_NATIVE_DIR")
    if not native_dir:
        raise RuntimeError(
            "set NPUNLOCK_NATIVE_DIR to the directory containing the built "
            "npunlock DLLs"
        )

    # npu.compile() reads NPUNLOCK_MOVITOOLS_DIR automatically. Alternatively,
    # configure the same directory in Python before compiling:
    # npu.configure(movi_dll_dir=r"D:\path\containing\MoviTools\DLLs")
    x = npu.input("x", shape=(1, 1024), dtype="f16")
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

    input_value = np.linspace(-4, 4, 1024, dtype=np.float16).reshape(1, 1024)
    outputs = program.run({"x": input_value})
    reference = gelu_reference(input_value)
    max_abs_error = np.max(
        np.abs(outputs["y"].astype(np.float32) - reference)
    )
    print(f"maximum absolute error: {max_abs_error:g}")


if __name__ == "__main__":
    main()
