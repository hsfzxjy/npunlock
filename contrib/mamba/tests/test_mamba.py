from __future__ import annotations

import unittest

import numpy as np

from mamba_npu.packing import (
    INVOCATION_WORDS,
    LOGICAL_WORDS,
    REPLICAS,
    pack_dynamic,
    pack_model,
    unpack_output,
)
from mamba_npu.reference import selective_scan


class MambaSmokeTests(unittest.TestCase):
    @staticmethod
    def make_case(sequence_length: int = 9):
        rng = np.random.default_rng(20261025)
        channels = 7
        state_size = 5
        u = rng.normal(0.0, 0.4, (channels, sequence_length)).astype(np.float32)
        delta = rng.normal(-2.0, 0.6, (channels, sequence_length)).astype(np.float32)
        a = -np.exp(rng.normal(0.0, 0.4, (channels, state_size))).astype(np.float32)
        b = rng.normal(0.0, 0.3, (sequence_length, state_size)).astype(np.float32)
        c = rng.normal(0.0, 0.3, (sequence_length, state_size)).astype(np.float32)
        d = rng.normal(0.0, 0.2, channels).astype(np.float32)
        z = rng.normal(0.0, 0.5, (channels, sequence_length)).astype(np.float32)
        bias = rng.normal(-3.0, 0.5, channels).astype(np.float32)
        state = rng.normal(0.0, 0.1, (channels, state_size)).astype(np.float32)
        return u, delta, a, b, c, d, z, bias, state

    def test_chunked_and_incremental_state_match_one_shot(self):
        u, delta, a, b, c, d, z, bias, state = self.make_case()
        expected, expected_state = selective_scan(u, delta, a, b, c, d, z, bias, state)
        first, carried = selective_scan(u[:, :4], delta[:, :4], a, b[:4], c[:4], d, z[:, :4], bias, state)
        second, chunked_state = selective_scan(u[:, 4:], delta[:, 4:], a, b[4:], c[4:], d, z[:, 4:], bias, carried)
        incremental_state = state
        incremental = []
        for index in range(u.shape[1]):
            output, incremental_state = selective_scan(
                u[:, index : index + 1],
                delta[:, index : index + 1],
                a,
                b[index : index + 1],
                c[index : index + 1],
                d,
                z[:, index : index + 1],
                bias,
                incremental_state,
            )
            incremental.append(output)
        np.testing.assert_allclose(np.concatenate((first, second), axis=1), expected, rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(np.concatenate(incremental, axis=1), expected, rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(chunked_state, expected_state, rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(incremental_state, expected_state, rtol=2e-6, atol=2e-6)

    def test_transport_round_trip_preserves_sequence_and_state(self):
        u, delta, a, b, c, d, z, bias, state = self.make_case(sequence_length=6)
        model, model_layout = pack_model(a, d, bias, sequence_length=u.shape[1])
        dynamic, layout = pack_dynamic(u, delta, b, c, z, state)
        self.assertEqual(model_layout, layout)
        self.assertEqual(model.shape, (1, LOGICAL_WORDS))
        self.assertEqual(dynamic.shape, (1, LOGICAL_WORDS))
        fake_output = np.zeros((REPLICAS, INVOCATION_WORDS), dtype=np.float32)
        words = fake_output.view(np.uint32)
        expected_sequence = np.arange(u.size, dtype=np.float32).reshape(u.shape)
        expected_state = np.arange(state.size, dtype=np.float32).reshape(state.shape)
        for replica, channel_slice in enumerate(layout.channel_slices):
            count = channel_slice.stop - channel_slice.start
            words[replica, :6] = (0x4D414D31, 1, 0, count, layout.state_size, layout.sequence_length)
            cursor = 16
            for channel in range(channel_slice.start, channel_slice.stop):
                fake_output[replica, cursor : cursor + layout.sequence_length] = expected_sequence[channel]
                cursor += layout.sequence_length
                fake_output[replica, cursor : cursor + layout.state_size] = expected_state[channel]
                cursor += layout.state_size
        sequence, final_state = unpack_output(fake_output, layout)
        np.testing.assert_array_equal(sequence, expected_sequence)
        np.testing.assert_array_equal(final_state, expected_state)


if __name__ == "__main__":
    unittest.main()
