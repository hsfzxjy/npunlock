"""Ordinary I8 Squeeze/Unsqueeze graph with a parallel FP32 output."""

import numpy as np
import npunlock as npu


def main() -> None:
    x = npu.input("x", shape=(1, 2, 1, 3), dtype="i8")
    axes = npu.constant(np.asarray([0, 2], dtype=np.int64), name="axes")
    squeezed = npu.Squeeze(x, axes, _name="squeezed_i8")
    restored = npu.Unsqueeze(squeezed, axes, _name="restored_i8")
    converted = npu.Convert(squeezed, destination_type="f32", _name="converted_f32")
    program = npu.compile(npu.Graph([x], [squeezed, restored, converted], name="i8_axes"))

    input_value = np.asarray([[[[-128, -127, -1]], [[0, 1, 127]]]], dtype=np.int8)
    expected = input_value.squeeze((0, 2))
    outputs = program.run({"x": input_value})

    np.testing.assert_array_equal(outputs["squeezed_i8"], expected)
    np.testing.assert_array_equal(outputs["restored_i8"], input_value)
    np.testing.assert_array_equal(outputs["converted_f32"], expected.astype(np.float32))
    print("I8 axis transforms and FP32 Convert output matched exact NumPy oracles")


if __name__ == "__main__":
    main()
