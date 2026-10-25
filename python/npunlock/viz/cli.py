from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

from .model import ArtifactGraph
from .npunlock import build_npunlock_graph
from .server import ViewerSession, show


def _port(value: str) -> int:
    try:
        port = int(value, 10)
    except ValueError as error:
        raise argparse.ArgumentTypeError("port must be a decimal integer") from error
    if not 0 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 0 and 65535")
    return port


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="npunlock-viz",
        description="Open one or two ELF artifacts in the read-only npunlock semantic hex viewer.",
    )
    parser.add_argument(
        "elf",
        nargs="+",
        metavar="ELF",
        type=Path,
        help="primary/native ELF followed by an optional standalone SHAVE kernel ELF",
    )
    parser.add_argument("--title", help="workspace title; defaults to the input file names")
    parser.add_argument("--port", type=_port, default=0, help="loopback port (default: choose an available port)")
    parser.add_argument("--no-browser", action="store_true", help="serve without opening the default browser")
    return parser


def _build_graph(paths: Sequence[Path], title: str | None) -> ArtifactGraph:
    primary = paths[0]
    kernel = paths[1] if len(paths) == 2 else None
    return build_npunlock_graph(
        primary,
        kernel,
        title=title or " · ".join(path.name for path in paths),
        native_name=primary.name,
        kernel_name=kernel.name if kernel is not None else "SHAVE kernel",
    )


def main(
    arguments: Sequence[str] | None = None,
    *,
    _show: Callable[..., ViewerSession] = show,
) -> int:
    parser = _parser()
    options = parser.parse_args(arguments)
    if len(options.elf) > 2:
        parser.error("expected a primary/native ELF and at most one standalone SHAVE kernel ELF")
    try:
        graph = _build_graph(options.elf, options.title)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    try:
        session = _show(graph, block=False, open_browser=not options.no_browser, port=options.port)
    except OSError as error:
        parser.exit(1, f"npunlock-viz: error: {error}\n")
    print(session.url, flush=True)
    try:
        session.wait()
    except KeyboardInterrupt:
        pass
    finally:
        session.close()
    return 0


__all__ = ["main"]
