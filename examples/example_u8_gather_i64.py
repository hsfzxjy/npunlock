"""Ordinary U8 Gather graph with caller-provided I64 indices."""

import numpy as np
import npunlock as npu


def main() -> None:
    data = npu.input("data", shape=(3, 4), dtype="u8")
    indices = npu.input("indices", shape=(2,), dtype="i64")
    axis = npu.constant(np.asarray(0, dtype=np.int64), name="axis")
    gathered = npu.Gather(data, indices, axis, _name="gathered")
    program = npu.compile(npu.Graph([data, indices], [gathered], name="u8_gather_i64"))

    data_value = np.asarray(
        [[0, 1, 254, 255], [7, 10, 40, 128], [255, 42, 0, 200]],
        dtype=np.uint8,
    )
    indices_value = np.asarray([2, 0], dtype=np.int64)
    expected = np.take(data_value, indices_value, axis=0)

    shared_inputs = program.shared_inputs()
    shared_output = program.shared_output()
    np.copyto(shared_inputs["data"], data_value)
    np.copyto(shared_inputs["indices"], indices_value)
    actual = program.run(shared_inputs, outputs={"gathered": shared_output})["gathered"]

    np.testing.assert_array_equal(actual, expected)
    print("shared U8/I64 Gather matched numpy.take exactly")


if __name__ == "__main__":
    main()
