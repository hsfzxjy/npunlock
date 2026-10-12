"""Ordinary U8 Reshape graph with U8 and FP32 outputs."""

import numpy as np
import npunlock as npu


def main() -> None:
    x = npu.input("x", shape=(2, 3), dtype="u8")
    pattern = npu.constant(np.asarray([3, -1], dtype=np.int64), name="pattern")
    reshaped = npu.Reshape(x, pattern, _name="reshaped_u8")
    converted = npu.Convert(reshaped, destination_type="f32", _name="converted_f32")
    program = npu.compile(npu.Graph([x], [reshaped, converted], name="u8_reshape"))

    input_value = np.asarray([[0, 1, 127], [128, 254, 255]], dtype=np.uint8)
    expected = input_value.reshape(3, 2)
    outputs = program.run({"x": input_value})

    np.testing.assert_array_equal(outputs["reshaped_u8"], expected)
    np.testing.assert_array_equal(outputs["converted_f32"], expected.astype(np.float32))
    print("U8 Reshape and FP32 Convert outputs matched exact NumPy oracles")


if __name__ == "__main__":
    main()
