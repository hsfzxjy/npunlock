"""Read-only semantic binary artifact visualization."""

from typing import TYPE_CHECKING, Any

from .model import Address, Artifact, ArtifactGraph, Category, Field, Link, Node, Span
from .server import ViewerSession, show

if TYPE_CHECKING:
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
    "show",
]


def __getattr__(name: str) -> Any:
    if name == "build_npunlock_graph":
        from .npunlock import build_npunlock_graph

        return build_npunlock_graph
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
