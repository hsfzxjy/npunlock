from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files
from typing import Any

from ._schema import validate_schema


BUNDLED_SCHEMAS = frozenset({"elf.json", "npu3720-act-compiler-8.3.json"})


@lru_cache(maxsize=None)
def load_bundled_schema(name: str) -> dict[str, Any]:
    """Load one reviewed, non-executable descriptor bundled with npunlock."""

    if name not in BUNDLED_SCHEMAS:
        raise ValueError(f"unknown bundled visualization schema: {name}")
    resource = files("npunlock.viz").joinpath("schemas", name)
    try:
        value = json.loads(resource.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load bundled visualization schema {name!r}") from error
    return validate_schema(value)


__all__ = ["BUNDLED_SCHEMAS", "load_bundled_schema"]
