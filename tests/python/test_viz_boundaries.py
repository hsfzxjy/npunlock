"""Offline dependency-boundary tests for the semantic artifact viewer."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "python" / "npunlock"
VIZ = PACKAGE / "viz"


def imported_modules(path: Path) -> set[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[tuple[int, str]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update((0, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add((node.level, node.module or ""))
    return imports


def test_generic_modules_do_not_import_adapter_or_core() -> None:
    generic_modules = (VIZ / "model.py", VIZ / "server.py", VIZ / "_schema.py", VIZ / "evidence.py")
    forbidden = {
        "npunlock._native",
        "npunlock.cache",
        "npunlock.contracts",
        "npunlock.external",
        "npunlock.graph",
        "npunlock.ir",
        "npunlock.kernel",
        "npunlock.tensor",
        "npunlock.viz.npunlock",
    }
    for path in generic_modules:
        imports = imported_modules(path)
        absolute = {name for level, name in imports if level == 0}
        relative = {name for level, name in imports if level > 0}
        assert not (absolute & forbidden), (path, absolute & forbidden)
        assert "npunlock" not in relative, (path, relative)


def test_core_does_not_import_visualizer() -> None:
    for path in PACKAGE.rglob("*.py"):
        if VIZ in path.parents:
            continue
        for level, name in imported_modules(path):
            assert not (level == 0 and (name == "npunlock.viz" or name.startswith("npunlock.viz."))), path
            assert not (level == 1 and (name == "viz" or name.startswith("viz."))), path


def test_generic_public_import_keeps_adapter_lazy() -> None:
    code = (
        "import sys; import npunlock.viz as viz; "
        "assert 'npunlock.viz.npunlock' not in sys.modules; "
        "assert 'npunlock.viz.evidence' not in sys.modules; "
        "assert viz.ArtifactGraph.__module__ == 'npunlock.viz.model'; "
        "assert 'npunlock.viz.npunlock' not in sys.modules; "
        "assert callable(viz.validate_evidence); "
        "assert 'npunlock.viz.evidence' in sys.modules; "
        "assert 'npunlock.viz.npunlock' not in sys.modules; "
        "assert callable(viz.build_npunlock_graph); "
        "assert 'npunlock.viz.npunlock' in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "python")},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def main() -> None:
    test_generic_modules_do_not_import_adapter_or_core()
    test_core_does_not_import_visualizer()
    test_generic_public_import_keeps_adapter_lazy()
    print("visualizer boundary tests passed")


if __name__ == "__main__":
    main()
