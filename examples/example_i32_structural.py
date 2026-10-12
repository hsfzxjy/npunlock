"""Ordinary multi-input/multi-output graph with structural ops and I32 tensors."""

import numpy as np
import npunlock as npu


def main() -> None:
    left = npu.input("left", shape=(2, 3), dtype="i32")
    right = npu.input("right", shape=(1, 3), dtype="i32")
    joined = npu.Concat(left, right, axis=0)
    order = npu.constant(np.asarray([1, 0], dtype=np.int64), name="order")
    transposed = npu.Transpose(joined, order, _name="transposed_i32")
    converted = npu.Convert(transposed, destination_type="f32", _name="converted_f32")
    program = npu.compile(npu.Graph([left, right], [transposed, converted], name="i32_structural"))

    left_value = np.asarray([[-7, 2, -9], [4, -5, 6]], dtype=np.int32)
    right_value = np.asarray([[-11, 12, -13]], dtype=np.int32)
    expected_i32 = np.concatenate([left_value, right_value], axis=0).T
    outputs = program.run({"left": left_value, "right": right_value})

    np.testing.assert_array_equal(outputs["transposed_i32"], expected_i32)
    np.testing.assert_array_equal(outputs["converted_f32"], expected_i32.astype(np.float32))
    print("I32 Concat/Transpose and FP32 Convert outputs matched exact host oracles")


if __name__ == "__main__":
    main()
