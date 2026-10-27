from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import npunlock as npu

from .packing import INVOCATION_WORDS, LOGICAL_WORDS, REPLICAS, pack_dynamic, pack_model, unpack_output
from .packing import MAX_SEQUENCE


KERNEL_PATH = Path(__file__).resolve().parents[1] / "kernels" / "selective_scan.c"


class NpuSelectiveScan:
    """One reusable NPU3720 selective-scan graph with explicit recurrent state I/O."""

    def __init__(
        self,
        a: object,
        d: object,
        delta_bias: object,
        *,
        movi_dll_dir: str | Path,
        native_dir: str | Path | None = None,
        timeout_ms: int = 20_000,
    ):
        a_value = np.asarray(a, dtype=np.float32)
        self.model_input, _ = pack_model(a_value, d, delta_bias)
        self.channels, self.state_size = a_value.shape
        npu.configure(movi_dll_dir=movi_dll_dir)
        dynamic = npu.input("mamba_dynamic", shape=(1, LOGICAL_WORDS), dtype="f32")
        model = npu.input("mamba_model", shape=(1, LOGICAL_WORDS), dtype="f32")
        output = npu.custom(
            dynamic,
            model,
            source=KERNEL_PATH.read_bytes(),
            carrier="Subtract",
            _name="mamba_scan_output",
        )
        prepared = npu.prepare(
            npu.Graph([dynamic, model], [output], name="mamba1_selective_scan"),
            native_dir=native_dir,
            timeout_ms=timeout_ms,
        )
        prepared.require(
            npu.CarrierContract(
                (
                    npu.ActGroupContract(
                        ("f32", "f32"),
                        "f32",
                        (INVOCATION_WORDS, INVOCATION_WORDS),
                        INVOCATION_WORDS,
                        REPLICAS,
                    ),
                ),
                mapping_indices=((0,),),
            )
        )
        started = perf_counter()
        self.program = prepared.build(movi_dll_dir=movi_dll_dir)
        self.build_ms = (perf_counter() - started) * 1000.0
        self.carrier_explanation = prepared.explain()
        self.shared_inputs = self.program.shared_inputs()
        self.shared_output = self.program.shared_output()
        np.copyto(self.shared_inputs["mamba_model"], self.model_input)

    def load_model(self, a: object, d: object, delta_bias: object) -> float:
        packed, layout = pack_model(a, d, delta_bias)
        if layout.channels != self.channels or layout.state_size != self.state_size:
            raise ValueError("new model parameters do not match the compiled runtime contract")
        started = perf_counter()
        np.copyto(self.shared_inputs["mamba_model"], packed)
        self.model_input = packed
        return (perf_counter() - started) * 1000.0

    def run(
        self,
        u: object,
        delta: object,
        b: object,
        c: object,
        z: object,
        initial_state: object | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        packed, layout = pack_dynamic(u, delta, b, c, z, initial_state)
        if layout.channels != self.channels or layout.state_size != self.state_size:
            raise ValueError("runtime inputs do not match the compiled model contract")
        started = perf_counter()
        np.copyto(self.shared_inputs["mamba_dynamic"], packed)
        upload_ms = (perf_counter() - started) * 1000.0
        started = perf_counter()
        self.program.run(self.shared_inputs, outputs={"mamba_scan_output": self.shared_output})
        execution_wall_ms = (perf_counter() - started) * 1000.0
        started = perf_counter()
        sequence, state = unpack_output(self.shared_output, layout)
        download_ms = (perf_counter() - started) * 1000.0
        return (
            sequence,
            state,
            {
                "upload_ms": upload_ms,
                "execution_wall_ms": execution_wall_ms,
                "download_ms": download_ms,
            },
        )

    def run_sequence(
        self,
        u: object,
        delta: object,
        b: object,
        c: object,
        z: object,
        initial_state: object | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        u_value = np.asarray(u, dtype=np.float32)
        delta_value = np.asarray(delta, dtype=np.float32)
        b_value = np.asarray(b, dtype=np.float32)
        c_value = np.asarray(c, dtype=np.float32)
        z_value = np.asarray(z, dtype=np.float32)
        if u_value.ndim != 2:
            raise ValueError("u must have shape (channels, sequence_length)")
        state = initial_state
        outputs = []
        totals = {"upload_ms": 0.0, "execution_wall_ms": 0.0, "download_ms": 0.0, "submissions": 0.0}
        for begin in range(0, u_value.shape[1], MAX_SEQUENCE):
            end = min(begin + MAX_SEQUENCE, u_value.shape[1])
            output, state, timings = self.run(
                u_value[:, begin:end],
                delta_value[:, begin:end],
                b_value[begin:end],
                c_value[begin:end],
                z_value[:, begin:end],
                state,
            )
            outputs.append(output)
            for name in ("upload_ms", "execution_wall_ms", "download_ms"):
                totals[name] += timings[name]
            totals["submissions"] += 1.0
        return np.concatenate(outputs, axis=1), np.asarray(state), totals

    def close(self) -> None:
        self.program.close()

    def __enter__(self) -> "NpuSelectiveScan":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
