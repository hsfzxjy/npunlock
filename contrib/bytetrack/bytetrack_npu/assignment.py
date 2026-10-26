from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Protocol

import numpy as np


MAX_ROWS = 16
MAX_COLUMNS = 16
CHUNK_ELEMENTS = 288
GRAPH_REPLICAS = 4
ACT_TARGETS = 4
LOGICAL_ELEMENTS = GRAPH_REPLICAS * CHUNK_ELEMENTS
DESCRIPTOR_ELEMENTS = ACT_TARGETS * CHUNK_ELEMENTS
INPUT_MAGIC = 372041
OUTPUT_MAGIC = 372042
FORMAT_VERSION = 1
INPUT_FORBIDDEN_OFFSET = 8
INPUT_COST_OFFSET = 32
OUTPUT_ROW_ASSIGNMENT_OFFSET = 16
OUTPUT_COLUMN_ASSIGNMENT_OFFSET = 32
OUTPUT_MATCH_COST_OFFSET = 48
OUTPUT_AUGMENTED_COST_OFFSET = 49
OUTPUT_WORDS = 50
COLUMN_BITS = np.left_shift(np.uint32(1), np.arange(MAX_COLUMNS, dtype=np.uint32))


@dataclass(frozen=True, slots=True)
class AssignmentResult:
    matches: np.ndarray
    unmatched_rows: np.ndarray
    unmatched_columns: np.ndarray
    match_cost: float
    augmented_cost: float
    timings: dict[str, float]


class AssignmentBackend(Protocol):
    def assign(
        self,
        costs: object,
        threshold: float,
        *,
        forbidden: object | None = None,
    ) -> AssignmentResult: ...


def _normalize_problem(
    costs: object,
    threshold: float,
    forbidden: object | None,
) -> tuple[np.ndarray, np.ndarray, np.float32]:
    matrix = np.asarray(costs, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("assignment costs must be a two-dimensional matrix")
    rows, columns = matrix.shape
    if rows > MAX_ROWS or columns > MAX_COLUMNS:
        raise ValueError(
            f"assignment shape {matrix.shape!r} exceeds the supported "
            f"{MAX_ROWS}x{MAX_COLUMNS} capacity"
        )
    limit = np.float32(threshold)
    if not np.isfinite(limit) or limit < 0:
        raise ValueError("assignment threshold must be a finite non-negative FP32 value")
    if forbidden is None:
        mask = np.zeros(matrix.shape, dtype=np.bool_)
    else:
        mask = np.asarray(forbidden, dtype=np.bool_)
        if mask.shape != matrix.shape:
            raise ValueError("forbidden mask shape must match the cost matrix")
        mask = mask.copy()
    matrix = np.ascontiguousarray(matrix)
    mask |= ~np.isfinite(matrix)
    mask |= matrix > limit
    return matrix, np.ascontiguousarray(mask), limit


def _decode_assignments(
    row_to_column: np.ndarray,
    column_to_row: np.ndarray,
    costs: np.ndarray,
    threshold: np.float32,
    timings: dict[str, float],
) -> AssignmentResult:
    rows, columns = costs.shape
    matched_rows = np.flatnonzero(row_to_column >= 0).astype(np.int32)
    matches = np.empty((matched_rows.size, 2), dtype=np.int32)
    if matched_rows.size:
        matches[:, 0] = matched_rows
        matches[:, 1] = row_to_column[matched_rows]
    unmatched_rows = np.flatnonzero(row_to_column < 0).astype(np.int32)
    unmatched_columns = np.flatnonzero(column_to_row < 0).astype(np.int32)
    match_cost = float(
        np.sum(costs[matches[:, 0], matches[:, 1]], dtype=np.float64)
        if matches.size
        else 0.0
    )
    augmented_cost = match_cost + float(
        np.float64(rows + columns - 2 * len(matches)) * np.float64(threshold) * 0.5
    )
    return AssignmentResult(
        matches=matches,
        unmatched_rows=unmatched_rows,
        unmatched_columns=unmatched_columns,
        match_cost=match_cost,
        augmented_cost=augmented_cost,
        timings=timings,
    )


def _validate_solution(
    result: AssignmentResult,
    costs: np.ndarray,
    forbidden: np.ndarray,
    threshold: np.float32,
) -> None:
    rows, columns = costs.shape
    matches = result.matches
    if matches.ndim != 2 or matches.shape[1:] != (2,):
        raise RuntimeError("assignment backend returned a malformed match array")
    if len(np.unique(matches[:, 0])) != len(matches) or len(np.unique(matches[:, 1])) != len(matches):
        raise RuntimeError("assignment backend returned duplicate row or column indices")
    if matches.size:
        if (
            (matches[:, 0] < 0).any()
            or (matches[:, 0] >= rows).any()
            or (matches[:, 1] < 0).any()
            or (matches[:, 1] >= columns).any()
        ):
            raise RuntimeError("assignment backend returned an out-of-range match")
        if forbidden[matches[:, 0], matches[:, 1]].any():
            raise RuntimeError("assignment backend selected a forbidden association")
        if (costs[matches[:, 0], matches[:, 1]] > threshold).any():
            raise RuntimeError("assignment backend selected an above-threshold association")
    expected_rows = np.setdiff1d(np.arange(rows, dtype=np.int32), matches[:, 0], assume_unique=True)
    expected_columns = np.setdiff1d(
        np.arange(columns, dtype=np.int32), matches[:, 1], assume_unique=True
    )
    if not np.array_equal(result.unmatched_rows, expected_rows) or not np.array_equal(
        result.unmatched_columns, expected_columns
    ):
        raise RuntimeError("assignment backend returned inconsistent unmatched indices")


def scipy_assignment(
    costs: object,
    threshold: float,
    *,
    forbidden: object | None = None,
) -> AssignmentResult:
    """Independent CPU reference using the LAPJV-compatible augmented problem."""

    from scipy.optimize import linear_sum_assignment

    matrix, mask, limit = _normalize_problem(costs, threshold, forbidden)
    rows, columns = matrix.shape
    order = rows + columns
    started = perf_counter()
    if order == 0:
        row_to_column = np.empty(0, dtype=np.int32)
        column_to_row = np.empty(0, dtype=np.int32)
    else:
        augmented = np.full((order, order), float(limit) * 0.5, dtype=np.float64)
        augmented[rows:, columns:] = 0.0
        augmented[:rows, :columns] = matrix
        augmented[:rows, :columns][mask] = np.inf
        assigned_rows, assigned_columns = linear_sum_assignment(augmented)
        row_to_column = np.full(rows, -1, dtype=np.int32)
        column_to_row = np.full(columns, -1, dtype=np.int32)
        for row, column in zip(assigned_rows.tolist(), assigned_columns.tolist()):
            if row < rows and column < columns and not mask[row, column]:
                row_to_column[row] = column
                column_to_row[column] = row
    elapsed = (perf_counter() - started) * 1000.0
    result = _decode_assignments(
        row_to_column,
        column_to_row,
        matrix,
        limit,
        {"cpu_assignment_ms": elapsed},
    )
    _validate_solution(result, matrix, mask, limit)
    return result


class CpuLapjvAssignment:
    """ByteTrack's conventional LAPJV path, kept as a reproducible baseline."""

    def assign(
        self,
        costs: object,
        threshold: float,
        *,
        forbidden: object | None = None,
    ) -> AssignmentResult:
        import lap

        matrix, mask, limit = _normalize_problem(costs, threshold, forbidden)
        if matrix.shape[0] == 0 or matrix.shape[1] == 0:
            row_to_column = np.full(matrix.shape[0], -1, dtype=np.int32)
            column_to_row = np.full(matrix.shape[1], -1, dtype=np.int32)
            result = _decode_assignments(
                row_to_column,
                column_to_row,
                matrix,
                limit,
                {"cpu_lapjv_ms": 0.0},
            )
            _validate_solution(result, matrix, mask, limit)
            return result
        safe = matrix.astype(np.float64)
        safe[mask] = np.inf
        started = perf_counter()
        _, row_to_column, column_to_row = lap.lapjv(
            safe,
            extend_cost=True,
            cost_limit=float(limit),
        )
        elapsed = (perf_counter() - started) * 1000.0
        result = _decode_assignments(
            np.asarray(row_to_column, dtype=np.int32),
            np.asarray(column_to_row, dtype=np.int32),
            matrix,
            limit,
            {"cpu_lapjv_ms": elapsed},
        )
        _validate_solution(result, matrix, mask, limit)
        return result


def _pack_problem(
    matrix: np.ndarray,
    forbidden: np.ndarray,
    threshold: np.float32,
    output: np.ndarray | None = None,
) -> np.ndarray:
    if output is None:
        packed = np.empty((GRAPH_REPLICAS, CHUNK_ELEMENTS), dtype=np.float32)
    else:
        packed = np.asarray(output, dtype=np.float32).reshape(GRAPH_REPLICAS, CHUNK_ELEMENTS)
    chunk = packed[0]
    chunk.fill(0)
    words = chunk.view(np.uint32)
    words[0:8] = (
        INPUT_MAGIC,
        FORMAT_VERSION,
        matrix.shape[0],
        matrix.shape[1],
        threshold.view(np.uint32),
        0,
        MAX_ROWS,
        MAX_COLUMNS,
    )
    if matrix.shape[0] and matrix.shape[1]:
        words[INPUT_FORBIDDEN_OFFSET : INPUT_FORBIDDEN_OFFSET + matrix.shape[0]] = np.sum(
            forbidden.astype(np.uint32) * COLUMN_BITS[: matrix.shape[1]],
            axis=1,
            dtype=np.uint32,
        )
    cost_view = chunk[INPUT_COST_OFFSET : INPUT_COST_OFFSET + MAX_ROWS * MAX_COLUMNS].reshape(
        MAX_ROWS, MAX_COLUMNS
    )
    cost_view[: matrix.shape[0], : matrix.shape[1]] = matrix
    packed[1:] = chunk
    return packed.reshape(1, LOGICAL_ELEMENTS)


def _decode_npu_output(
    output: object,
    matrix: np.ndarray,
    forbidden: np.ndarray,
    threshold: np.float32,
    timings: dict[str, float],
) -> AssignmentResult:
    chunks = np.asarray(output, dtype=np.float32).reshape(GRAPH_REPLICAS, CHUNK_ELEMENTS)
    replica_words = chunks[:, :OUTPUT_WORDS].view(np.uint32)
    if not np.all(replica_words[1:] == replica_words[0]):
        raise RuntimeError("ACT assignment replicas returned different output bytes")
    chunk = chunks[0]
    words = chunk.view(np.uint32)
    rows, columns = matrix.shape
    if words[0] != OUTPUT_MAGIC or words[1] != FORMAT_VERSION:
        raise RuntimeError("ACT assignment output header is invalid")
    if words[2] != 0:
        raise RuntimeError(f"ACT assignment kernel rejected the input with status {int(words[2])}")
    if words[3] != rows or words[4] != columns:
        raise RuntimeError("ACT assignment output shape does not match the request")
    row_to_column = words[
        OUTPUT_ROW_ASSIGNMENT_OFFSET : OUTPUT_ROW_ASSIGNMENT_OFFSET + rows
    ].view(np.int32).tolist()
    column_to_row = words[
        OUTPUT_COLUMN_ASSIGNMENT_OFFSET : OUTPUT_COLUMN_ASSIGNMENT_OFFSET + columns
    ].view(np.int32).tolist()
    matches = []
    unmatched_rows = []
    unmatched_columns = []
    match_cost = 0.0
    for row, column in enumerate(row_to_column):
        if column == -1:
            unmatched_rows.append(row)
        elif (
            column < 0
            or column >= columns
            or column_to_row[column] != row
            or forbidden[row, column]
            or matrix[row, column] > threshold
        ):
            raise RuntimeError("ACT assignment row and column mappings are inconsistent")
        else:
            matches.append((row, column))
            match_cost += float(matrix[row, column])
    for column, row in enumerate(column_to_row):
        if row == -1:
            unmatched_columns.append(column)
        elif row < 0 or row >= rows or row_to_column[row] != column:
            raise RuntimeError("ACT assignment column and row mappings are inconsistent")
    match_array = np.asarray(matches, dtype=np.int32).reshape(-1, 2)
    augmented_cost = match_cost + float(
        np.float64(rows + columns - 2 * len(matches)) * np.float64(threshold) * 0.5
    )
    result = AssignmentResult(
        matches=match_array,
        unmatched_rows=np.asarray(unmatched_rows, dtype=np.int32),
        unmatched_columns=np.asarray(unmatched_columns, dtype=np.int32),
        match_cost=match_cost,
        augmented_cost=augmented_cost,
        timings=timings,
    )
    if words[5] != len(result.matches) or words[6] != len(result.unmatched_rows) or words[7] != len(
        result.unmatched_columns
    ):
        raise RuntimeError("ACT assignment output counters are inconsistent")
    if not np.isclose(chunk[OUTPUT_MATCH_COST_OFFSET], result.match_cost, rtol=2e-6, atol=2e-6):
        raise RuntimeError("ACT assignment output cost does not match its assignments")
    if not np.isclose(
        chunk[OUTPUT_AUGMENTED_COST_OFFSET], result.augmented_cost, rtol=2e-6, atol=2e-6
    ):
        raise RuntimeError("ACT assignment augmented cost does not match its assignments")
    return result


class NpuAssignment:
    """Bounded optimal assignment executed by a custom NPU3720 ACT kernel."""

    def __init__(
        self,
        *,
        movi_dll_dir: str | Path,
        native_dir: str | Path | None = None,
        timeout_ms: int = 20_000,
        kernel_path: str | Path | None = None,
        shared: bool = False,
    ):
        import npunlock as npu

        if kernel_path is None:
            kernel_path = Path(__file__).resolve().parents[1] / "kernels" / "assignment.c"
        source = Path(kernel_path).read_bytes()
        npu.configure(movi_dll_dir=movi_dll_dir)
        payload = npu.input("assignment_payload", shape=(1, LOGICAL_ELEMENTS), dtype="f32")
        guard = npu.input("assignment_guard", shape=(1, LOGICAL_ELEMENTS), dtype="f32")
        output = npu.custom(
            payload,
            guard,
            source=source,
            carrier="Subtract",
            _name="assignment_result",
        )
        prepared = npu.prepare(
            npu.Graph([payload, guard], [output], name="bytetrack_linear_assignment"),
            native_dir=native_dir,
            timeout_ms=timeout_ms,
        )
        if len(prepared.groups) != 1:
            raise RuntimeError(
                f"assignment carrier exposed {len(prepared.groups)} ACT groups, expected one"
            )
        group = prepared.groups[0]
        if (
            group.input_dtypes != ("f32", "f32")
            or group.output_dtype != "f32"
            or group.element_count != DESCRIPTOR_ELEMENTS
            or group.per_invocation_input_element_counts != (CHUNK_ELEMENTS, CHUNK_ELEMENTS)
            or group.per_invocation_output_element_count != CHUNK_ELEMENTS
            or len(group.targets) != ACT_TARGETS
        ):
            raise RuntimeError(
                "assignment carrier contract changed: "
                f"inputs={group.per_invocation_input_element_counts!r}, "
                f"output={group.per_invocation_output_element_count!r}, "
                f"targets={len(group.targets)}"
            )
        started = perf_counter()
        self.program = prepared.build(movi_dll_dir=movi_dll_dir)
        self.build_ms = (perf_counter() - started) * 1000.0
        self.carrier_explanation = prepared.explain()
        self.shared = bool(shared)
        self.guard = np.zeros((1, LOGICAL_ELEMENTS), dtype=np.float32)
        self.shared_inputs = self.program.shared_inputs() if self.shared else None
        if self.shared_inputs is not None:
            self.shared_input = self.shared_inputs["assignment_payload"]
            self.shared_guard = self.shared_inputs["assignment_guard"]
            self.shared_guard.fill(0)
        else:
            self.shared_input = None
            self.shared_guard = None
        self.packed_input = (
            self.shared_input
            if self.shared_input is not None
            else np.empty((1, LOGICAL_ELEMENTS), dtype=np.float32)
        )
        self.shared_output = self.program.shared_output() if self.shared else None
        self.target_count = len(group.targets)
        self.source_sha256 = hashlib.sha256(source).hexdigest()

    @property
    def patch_evidence(self) -> dict[str, object]:
        if len(self.program.patch_reports) != 1:
            raise RuntimeError("assignment program has an unexpected patch-report count")
        return json.loads(self.program.patch_reports[0].decode("utf-8"))

    def assign(
        self,
        costs: object,
        threshold: float,
        *,
        forbidden: object | None = None,
    ) -> AssignmentResult:
        matrix, mask, limit = _normalize_problem(costs, threshold, forbidden)
        fast_path = None
        if matrix.shape[0] == 0 or matrix.shape[1] == 0:
            fast_path = "empty_fast_path_ms"
        elif mask.all():
            fast_path = "infeasible_fast_path_ms"
        if fast_path is not None:
            started = perf_counter()
            rows, columns = matrix.shape
            return AssignmentResult(
                matches=np.empty((0, 2), dtype=np.int32),
                unmatched_rows=np.arange(rows, dtype=np.int32),
                unmatched_columns=np.arange(columns, dtype=np.int32),
                match_cost=0.0,
                augmented_cost=float(
                    np.float64(rows + columns) * np.float64(limit) * 0.5
                ),
                timings={fast_path: (perf_counter() - started) * 1000.0},
            )
        started = perf_counter()
        # The cost-limited objective is symmetric; making the smaller side the
        # row side reduces the number of augmentations for tall matrices.
        transposed = matrix.shape[0] > matrix.shape[1]
        device_matrix = np.ascontiguousarray(matrix.T) if transposed else matrix
        device_mask = np.ascontiguousarray(mask.T) if transposed else mask
        packed = _pack_problem(device_matrix, device_mask, limit, self.packed_input)
        pack_or_upload_ms = (perf_counter() - started) * 1000.0
        started = perf_counter()
        payload = packed if self.shared_input is None else self.shared_input
        guard = self.guard if self.shared_guard is None else self.shared_guard
        outputs = self.program.run(
            {
                "assignment_payload": payload,
                "assignment_guard": guard,
            },
            outputs=None if self.shared_output is None else {"assignment_result": self.shared_output},
        )
        graphinfer_wall_ms = (perf_counter() - started) * 1000.0
        started = perf_counter()
        result = _decode_npu_output(
            outputs["assignment_result"],
            device_matrix,
            device_mask,
            limit,
            {},
        )
        if transposed:
            matches = result.matches[:, ::-1].copy()
            if len(matches) > 1:
                matches = matches[np.argsort(matches[:, 0], kind="stable")]
            result = AssignmentResult(
                matches=matches,
                unmatched_rows=result.unmatched_columns.copy(),
                unmatched_columns=result.unmatched_rows.copy(),
                match_cost=result.match_cost,
                augmented_cost=result.augmented_cost,
                timings={},
            )
        decode_ms = (perf_counter() - started) * 1000.0
        return AssignmentResult(
            matches=result.matches,
            unmatched_rows=result.unmatched_rows,
            unmatched_columns=result.unmatched_columns,
            match_cost=result.match_cost,
            augmented_cost=result.augmented_cost,
            timings={
                "pack_or_upload_ms": pack_or_upload_ms,
                "graphinfer_wall_ms": graphinfer_wall_ms,
                "decode_ms": decode_ms,
            },
        )

    def close(self) -> None:
        self.program.close()

    def __enter__(self) -> "NpuAssignment":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
