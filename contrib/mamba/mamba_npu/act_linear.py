from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import npunlock as npu


MAGIC = 0x4C494E31
VERSION = 1
REPLICAS = 4
INVOCATION_WORDS = 65536
LOGICAL_WORDS = REPLICAS * INVOCATION_WORDS
HEADER_WORDS = 16
KERNEL_PATH = Path(__file__).resolve().parents[1] / "kernels" / "linear_f32.c"


class NpuActLinear:
    def __init__(self, *, movi_dll_dir: str | Path, native_dir: str | Path | None = None):
        activation = npu.input("linear_activation", shape=(1, LOGICAL_WORDS), dtype="f32")
        weights = npu.input("linear_weights", shape=(1, LOGICAL_WORDS), dtype="f32")
        output = npu.custom(
            activation,
            weights,
            source=KERNEL_PATH.read_bytes(),
            carrier="Subtract",
            _name="linear_output",
        )
        started = perf_counter()
        self.program = npu.compile(
            npu.Graph([activation, weights], [output], name="mamba_act_linear_f32"),
            movi_dll_dir=movi_dll_dir,
            native_dir=native_dir,
        )
        self.build_ms = (perf_counter() - started) * 1000.0
        self.inputs = self.program.shared_inputs()
        self.output = self.program.shared_output()

    def run(self, activation: object, weight: object) -> tuple[np.ndarray, dict[str, float]]:
        vector = np.asarray(activation, dtype=np.float32).reshape(-1)
        matrix = np.asarray(weight, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != vector.size or not 1 <= vector.size <= 1536:
            raise ValueError("weight must have shape (input_features, output_features)")
        capacity = (INVOCATION_WORDS - HEADER_WORDS) // vector.size
        per_submission = capacity * REPLICAS
        result = np.empty(matrix.shape[1], dtype=np.float32)
        total_execution = total_upload = total_download = 0.0
        submissions = 0
        for begin in range(0, matrix.shape[1], per_submission):
            activation_pack = np.zeros((REPLICAS, INVOCATION_WORDS), dtype=np.float32)
            weight_pack = np.zeros_like(activation_pack)
            awords = activation_pack.view(np.uint32)
            wwords = weight_pack.view(np.uint32)
            counts = []
            for replica in range(REPLICAS):
                replica_begin = begin + replica * capacity
                count = max(0, min(capacity, matrix.shape[1] - replica_begin))
                counts.append(count)
                awords[replica, :4] = (MAGIC, VERSION, vector.size, count)
                wwords[replica, :4] = (MAGIC, VERSION, vector.size, count)
                activation_pack[replica, HEADER_WORDS : HEADER_WORDS + vector.size] = vector
                if count:
                    weight_pack[replica, HEADER_WORDS : HEADER_WORDS + count * vector.size] = matrix[
                        :, replica_begin : replica_begin + count
                    ].T.reshape(-1)
            started = perf_counter()
            np.copyto(self.inputs["linear_activation"], activation_pack.reshape(1, -1))
            np.copyto(self.inputs["linear_weights"], weight_pack.reshape(1, -1))
            total_upload += (perf_counter() - started) * 1000.0
            started = perf_counter()
            self.program.run(self.inputs, outputs={"linear_output": self.output})
            total_execution += (perf_counter() - started) * 1000.0
            started = perf_counter()
            chunks = np.asarray(self.output).reshape(REPLICAS, INVOCATION_WORDS)
            words = chunks.view(np.uint32)
            for replica, count in enumerate(counts):
                if tuple(words[replica, :3]) != (MAGIC, VERSION, count):
                    raise RuntimeError("ACT linear output metadata is invalid")
                replica_begin = begin + replica * capacity
                result[replica_begin : replica_begin + count] = chunks[
                    replica, HEADER_WORDS : HEADER_WORDS + count
                ]
            total_download += (perf_counter() - started) * 1000.0
            submissions += 1
        return result.reshape(1, -1), {
            "upload_ms": total_upload,
            "execution_wall_ms": total_execution,
            "download_ms": total_download,
            "submissions": float(submissions),
        }

    def close(self) -> None:
        self.program.close()

    def __enter__(self) -> "NpuActLinear":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
