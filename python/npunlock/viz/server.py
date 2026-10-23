from __future__ import annotations

import json
import mimetypes
import secrets
import threading
import urllib.parse
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from typing import Callable

from .model import ArtifactGraph


MAX_RANGE_BYTES = 64 * 1024


class _ViewerHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, address: tuple[str, int], graph: ArtifactGraph, token: str) -> None:
        self.graph = graph
        self.token = token
        self.static_root = files("npunlock.viz").joinpath("static")
        super().__init__(address, _ViewerHandler)


class _ViewerHandler(BaseHTTPRequestHandler):
    server: _ViewerHTTPServer

    def log_message(self, _format: str, *args: object) -> None:
        return

    def _headers(self, content_type: str, length: int, *, cache: str = "no-store") -> None:
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
            "connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
        )

    def _send_bytes(
        self,
        body: bytes,
        content_type: str,
        *,
        status: HTTPStatus = HTTPStatus.OK,
        cache: str = "no-store",
    ) -> None:
        self.send_response(status)
        self._headers(content_type, len(body), cache=cache)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _error(self, status: HTTPStatus, message: str) -> None:
        body = json.dumps({"error": message}, separators=(",", ":")).encode("utf-8")
        self._send_bytes(body, "application/json; charset=utf-8", status=status)

    def do_POST(self) -> None:
        self._error(HTTPStatus.METHOD_NOT_ALLOWED, "viewer endpoints are read-only")

    do_PUT = do_POST
    do_PATCH = do_POST
    do_DELETE = do_POST

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_GET(self) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        prefix = f"/{self.server.token}/"
        if not parsed.path.startswith(prefix):
            self._error(HTTPStatus.NOT_FOUND, "not found")
            return
        relative = parsed.path[len(prefix) :]
        if relative == "api/graph":
            body = json.dumps(
                self.server.graph.to_dict(include_data=False),
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            self._send_bytes(body, "application/json; charset=utf-8")
            return
        if relative.startswith("api/artifacts/") and relative.endswith("/bytes"):
            encoded_id = relative[len("api/artifacts/") : -len("/bytes")]
            artifact_id = urllib.parse.unquote(encoded_id)
            artifact = self.server.graph.artifacts.get(artifact_id)
            if artifact is None:
                self._error(HTTPStatus.NOT_FOUND, "unknown artifact")
                return
            try:
                query = urllib.parse.parse_qs(parsed.query, strict_parsing=True)
                offset = int(query.get("offset", ["0"])[0], 10)
                length = int(query.get("length", [str(MAX_RANGE_BYTES)])[0], 10)
            except (ValueError, TypeError):
                self._error(HTTPStatus.BAD_REQUEST, "offset and length must be decimal integers")
                return
            if offset < 0 or length < 0 or length > MAX_RANGE_BYTES or offset > artifact.size:
                self._error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, "invalid or excessive byte range")
                return
            end = min(artifact.size, offset + length)
            try:
                body = artifact.read(offset, end)
            except OSError as error:
                self._error(HTTPStatus.CONFLICT, str(error))
                return
            self._send_bytes(body, "application/octet-stream")
            return
        if relative in {"", "index.html"}:
            relative = "index.html"
        if relative not in {"index.html", "app.js", "app.css"}:
            self._error(HTTPStatus.NOT_FOUND, "not found")
            return
        resource = self.server.static_root.joinpath(relative)
        try:
            body = resource.read_bytes()
        except (FileNotFoundError, OSError):
            self._error(HTTPStatus.NOT_FOUND, "static asset not found")
            return
        content_type = mimetypes.guess_type(relative)[0] or "application/octet-stream"
        if content_type.startswith("text/") or relative.endswith(".js"):
            content_type += "; charset=utf-8"
        self._send_bytes(body, content_type, cache="public, max-age=3600")


class ViewerSession:
    """A running loopback viewer server. Call ``close`` when finished."""

    def __init__(self, graph: ArtifactGraph, *, port: int = 0) -> None:
        graph.validate()
        self._server = _ViewerHTTPServer(("127.0.0.1", port), graph, secrets.token_urlsafe(18))
        self._thread = threading.Thread(target=self._server.serve_forever, name="npunlock-viz", daemon=True)
        self._closed = False
        self._thread.start()

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/{self._server.token}/"

    @property
    def closed(self) -> bool:
        return self._closed

    def wait(self) -> None:
        # An indefinite Thread.join() is not interruptible by Ctrl+C on the
        # supported Windows/Python 3.12 host. Timed joins let the main thread
        # regularly dispatch KeyboardInterrupt while remaining inexpensive.
        while self._thread.is_alive():
            self._thread.join(timeout=0.2)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._server.shutdown()
        self._server.server_close()
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=5)

    def __enter__(self) -> ViewerSession:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def show(
    graph: ArtifactGraph,
    *,
    block: bool = True,
    open_browser: bool = True,
    port: int = 0,
    _browser_open: Callable[[str], object] = webbrowser.open,
) -> ViewerSession:
    """Serve ``graph`` on loopback and open it in the default browser.

    The default blocks until Ctrl+C, which gives normal scripts a deterministic
    lifetime. Pass ``block=False`` in notebooks or interactive shells and close
    the returned session explicitly (or use it as a context manager).
    """

    session = ViewerSession(graph, port=port)
    try:
        if open_browser:
            _browser_open(session.url)
    except BaseException:
        session.close()
        raise
    if block:
        try:
            session.wait()
        except KeyboardInterrupt:
            session.close()
    return session
