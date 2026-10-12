from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path


def _kernel_source(value: object) -> bytes:
    if isinstance(value, bytes):
        source = value
    elif isinstance(value, (str, os.PathLike)):
        source = Path(value).read_bytes()
    else:
        raise TypeError("custom kernel source must be bytes or a filesystem path")
    if not source:
        raise ValueError("custom kernel source must not be empty")
    return source


def _linker_script(value: object) -> bytes | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value or None
    if isinstance(value, (str, os.PathLike)):
        script = Path(value).read_bytes()
        return script or None
    raise TypeError("linker_script must be bytes, a filesystem path, or None")


def _compiler_definitions(values: Sequence[str]) -> tuple[str, ...]:
    definitions = tuple(values)
    if len(definitions) > 64:
        raise ValueError("at most 64 compiler definitions are supported")
    for definition in definitions:
        if not isinstance(definition, str):
            raise TypeError("compiler definitions must be strings")
        name, separator, value = definition.partition("=")
        if (
            separator != "="
            or not name
            or not ("A" <= name[0] <= "Z")
            or any(not ("A" <= character <= "Z" or "0" <= character <= "9" or character == "_") for character in name)
            or not value
            or any(not "0" <= character <= "9" for character in value)
            or len(definition.encode("utf-8")) > 255
        ):
            raise ValueError("compiler definitions must use the uppercase NAME=DECIMAL form")
    return definitions


@dataclass(frozen=True, slots=True)
class KernelSpec:
    """A snapshotted C source and its per-kernel compiler settings."""

    source: bytes
    definitions: tuple[str, ...]
    linker_script: bytes | None
    source_name: str = field(compare=False)

    def __init__(
        self,
        source: bytes | str | os.PathLike[str],
        *,
        definitions: Sequence[str] = (),
        linker_script: bytes | str | os.PathLike[str] | None = None,
    ) -> None:
        source_name = os.fspath(source) if isinstance(source, (str, os.PathLike)) else "<bytes>"
        object.__setattr__(self, "source", _kernel_source(source))
        object.__setattr__(self, "definitions", _compiler_definitions(definitions))
        object.__setattr__(self, "linker_script", _linker_script(linker_script))
        object.__setattr__(self, "source_name", source_name)

    @property
    def source_sha256(self) -> str:
        return hashlib.sha256(self.source).hexdigest()

    @property
    def linker_script_sha256(self) -> str | None:
        return None if self.linker_script is None else hashlib.sha256(self.linker_script).hexdigest()


@dataclass(frozen=True, slots=True)
class CompiledKernel:
    """A snapshotted, already linked SHAVE ELF."""

    elf: bytes
    elf_name: str = field(compare=False)

    def __init__(self, elf: bytes | str | os.PathLike[str]) -> None:
        elf_name = os.fspath(elf) if isinstance(elf, (str, os.PathLike)) else "<bytes>"
        if isinstance(elf, bytes):
            elf_bytes = elf
        elif isinstance(elf, (str, os.PathLike)):
            elf_bytes = Path(elf).read_bytes()
        else:
            raise TypeError("compiled kernel must be ELF bytes or a filesystem path")
        if not elf_bytes:
            raise ValueError("compiled kernel ELF must not be empty")
        object.__setattr__(self, "elf", elf_bytes)
        object.__setattr__(self, "elf_name", elf_name)

    @property
    def elf_sha256(self) -> str:
        return hashlib.sha256(self.elf).hexdigest()

    def save(self, destination: str | Path) -> None:
        """Write the snapshotted linked ELF to *destination*."""

        Path(destination).write_bytes(self.elf)
