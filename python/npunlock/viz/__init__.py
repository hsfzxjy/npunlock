"""Read-only semantic binary artifact visualization."""

from typing import TYPE_CHECKING, Any

from .model import Address, Artifact, ArtifactGraph, Category, Field, Link, Node, Span
from .server import ViewerSession, show

if TYPE_CHECKING:
    from .evidence import project_prepared_evidence, validate_evidence
    from .npunlock import build_npunlock_graph

__all__ = [
    "Address",
    "Artifact",
    "ArtifactGraph",
    "Category",
    "Field",
    "Link",
    "Node",
    "Span",
    "ViewerSession",
    "build_npunlock_graph",
    "project_prepared_evidence",
    "show",
    "validate_evidence",
]


def __getattr__(name: str) -> Any:
    if name == "build_npunlock_graph":
        from .npunlock import build_npunlock_graph

        return build_npunlock_graph
    if name in {"project_prepared_evidence", "validate_evidence"}:
        from .evidence import project_prepared_evidence, validate_evidence

        return {"project_prepared_evidence": project_prepared_evidence, "validate_evidence": validate_evidence}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
