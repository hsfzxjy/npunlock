from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from .graph import Node

Shape: TypeAlias = tuple[int, ...]
DType: TypeAlias = str

_DTYPES = {
    "boolean",
    "f16",
    "f32",
    "f64",
    "i8",
    "i16",
    "i32",
    "i64",
    "u8",
    "u16",
    "u32",
    "u64",
}


def normalize_shape(shape: object) -> Shape:
    if isinstance(shape, (str, bytes)):
        raise TypeError("shape must be an iterable of positive integers")
    try:
        normalized = tuple(shape)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError("shape must be an iterable of positive integers") from exc
    if any(isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0 for dim in normalized):
        raise ValueError("only static positive dimensions are supported")
    return normalized


def normalize_dtype(dtype: object) -> DType:
    value = str(dtype).lower()
    aliases = {"bool": "boolean", "float16": "f16", "float32": "f32", "float64": "f64"}
    value = aliases.get(value, value)
    if value not in _DTYPES:
        raise ValueError(f"unsupported IR element type: {dtype!r}")
    return value


@dataclass(frozen=True, slots=True)
class TensorSpec:
    shape: Shape
    dtype: DType

    def __init__(self, shape: object, dtype: object):
        object.__setattr__(self, "shape", normalize_shape(shape))
        object.__setattr__(self, "dtype", normalize_dtype(dtype))


@dataclass(frozen=True, slots=True, eq=False)
class Tensor:
    producer: Node
    output_index: int
    spec: TensorSpec
    name: str | None = None

    @property
    def shape(self) -> Shape:
        return self.spec.shape

    @property
    def dtype(self) -> DType:
        return self.spec.dtype

    def _binary(self, operator: str, other: Tensor) -> Tensor:
        from .graph import op

        if not isinstance(other, Tensor):
            return NotImplemented  # type: ignore[return-value]
        if self.spec != other.spec:
            raise ValueError(f"{operator} sugar requires matching symbolic tensor specs")
        return op(operator, self, other, _shape=self.shape, _dtype=self.dtype)

    def __add__(self, other: Tensor) -> Tensor:
        return self._binary("Add", other)

    def __mul__(self, other: Tensor) -> Tensor:
        return self._binary("Multiply", other)

    def __matmul__(self, other: Tensor) -> Tensor:
        return self._binary("MatMul", other)
