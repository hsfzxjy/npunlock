from __future__ import annotations

import base64
import hashlib
import json
import os
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


FORMAT = "npunlock.artifact-graph"
FORMAT_VERSION = 1


def _identifier(value: str, what: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{what} must be a non-empty string without surrounding whitespace")
    return value


def _json_value(value: Any, what: str) -> Any:
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{what} must be JSON-serializable") from error
    return value


@dataclass(frozen=True, slots=True)
class Category:
    id: str
    label: str
    color: str

    def __post_init__(self) -> None:
        _identifier(self.id, "category id")
        _identifier(self.label, "category label")
        if not isinstance(self.color, str) or not self.color.startswith("#") or len(self.color) not in {4, 7}:
            raise ValueError("category color must be a #rgb or #rrggbb string")


DEFAULT_CATEGORIES = (
    Category("metadata", "Headers and metadata", "#6f83a6"),
    Category("code", "Executable code", "#8a7652"),
    Category("data", "Parameters and data", "#587d70"),
    Category("reference", "References and relocations", "#7c668f"),
    Category("unknown", "Uninterpreted and padding", "#5e6670"),
)


@dataclass(frozen=True, slots=True)
class Address:
    """A non-file address kept explicitly separate from byte offsets."""

    space: str
    value: int

    def __post_init__(self) -> None:
        _identifier(self.space, "address space")
        if isinstance(self.value, bool) or not isinstance(self.value, int) or self.value < 0:
            raise ValueError("address value must be a non-negative integer")


class Artifact:
    """One immutable binary source, backed by bytes or a file path."""

    __slots__ = ("id", "name", "media_type", "size", "sha256", "_bytes", "_path", "_mtime_ns")

    def __init__(
        self,
        artifact_id: str,
        *,
        data: bytes | bytearray | memoryview | str | os.PathLike[str] | None = None,
        path: str | os.PathLike[str] | None = None,
        name: str | None = None,
        media_type: str = "application/octet-stream",
    ) -> None:
        self.id = _identifier(artifact_id, "artifact id")
        self.name = _identifier(name if name is not None else artifact_id, "artifact name")
        if not isinstance(media_type, str) or not media_type:
            raise ValueError("artifact media_type must be a non-empty string")
        self.media_type = media_type
        if data is not None and path is not None:
            raise ValueError("supply artifact data or path, not both")
        if path is None and isinstance(data, (str, os.PathLike)):
            path = data
            data = None
        if path is None and data is None:
            raise ValueError("artifact requires bytes or a file path")
        self._bytes: bytes | None = None
        self._path: Path | None = None
        self._mtime_ns: int | None = None
        digest = hashlib.sha256()
        if path is not None:
            resolved = Path(path).expanduser().resolve(strict=True)
            if not resolved.is_file():
                raise ValueError(f"artifact path is not a file: {resolved}")
            self._path = resolved
            source_stat = resolved.stat()
            self.size = source_stat.st_size
            self._mtime_ns = source_stat.st_mtime_ns
            with resolved.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
        else:
            assert data is not None
            self._bytes = bytes(data)
            self.size = len(self._bytes)
            digest.update(self._bytes)
        self.sha256 = digest.hexdigest()

    @property
    def source_kind(self) -> str:
        return "file" if self._path is not None else "memory"

    def read(self, start: int = 0, end: int | None = None) -> bytes:
        if isinstance(start, bool) or not isinstance(start, int) or start < 0:
            raise ValueError("start must be a non-negative integer")
        if end is None:
            end = self.size
        if isinstance(end, bool) or not isinstance(end, int) or end < start or end > self.size:
            raise ValueError("artifact read range is outside the artifact")
        if self._bytes is not None:
            return self._bytes[start:end]
        assert self._path is not None
        source_stat = self._path.stat()
        if source_stat.st_size != self.size or source_stat.st_mtime_ns != self._mtime_ns:
            raise OSError(f"artifact changed after graph construction: {self._path}")
        with self._path.open("rb") as stream:
            stream.seek(start)
            value = stream.read(end - start)
        if len(value) != end - start:
            raise OSError(f"artifact could not be read completely: {self._path}")
        return value

    def to_dict(self, *, include_data: bool = False) -> dict[str, Any]:
        value: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "size": self.size,
            "sha256": self.sha256,
            "media_type": self.media_type,
        }
        if include_data:
            value["data_base64"] = base64.b64encode(self.read()).decode("ascii")
        return value


@dataclass(frozen=True, slots=True)
class Span:
    artifact: Artifact
    start: int
    end: int

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, Artifact):
            raise TypeError("span artifact must be an Artifact")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (self.start, self.end)):
            raise TypeError("span offsets must be integers")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("span must be a non-empty half-open interval [start, end)")

    @property
    def size(self) -> int:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        return {"artifact": self.artifact.id, "start": self.start, "end": self.end}


@dataclass(frozen=True, slots=True)
class Field:
    value: Any
    spans: tuple[Span, ...] = ()
    display: str | None = None

    def __init__(self, value: Any, spans: Iterable[Span] = (), display: str | None = None) -> None:
        object.__setattr__(self, "value", _json_value(value, "field value"))
        object.__setattr__(self, "spans", tuple(spans))
        if display is not None and not isinstance(display, str):
            raise TypeError("field display must be a string or None")
        object.__setattr__(self, "display", display)

    def to_dict(self) -> dict[str, Any]:
        value = {"value": self.value}
        if self.display is not None:
            value["display"] = self.display
        if self.spans:
            value["spans"] = [span.to_dict() for span in self.spans]
        return value


@dataclass(frozen=True, slots=True)
class Node:
    id: str
    label: str
    kind: str
    category: str
    spans: tuple[Span, ...] = ()
    properties: Mapping[str, Any] = field(default_factory=dict)
    fields: Mapping[str, Field] = field(default_factory=dict)
    addresses: Mapping[str, Address] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Link:
    source: str
    target: str
    kind: str
    properties: Mapping[str, Any] = field(default_factory=dict)


class ArtifactGraph:
    """Format-agnostic semantic annotations over one or more binary artifacts."""

    def __init__(self, title: str, *, categories: Iterable[Category] = DEFAULT_CATEGORIES) -> None:
        self.title = _identifier(title, "graph title")
        self.artifacts: OrderedDict[str, Artifact] = OrderedDict()
        self.nodes: OrderedDict[str, Node] = OrderedDict()
        self.links: list[Link] = []
        self._parents: dict[str, str] = {}
        self.categories: OrderedDict[str, Category] = OrderedDict()
        for category in categories:
            if category.id in self.categories:
                raise ValueError(f"duplicate category id: {category.id}")
            self.categories[category.id] = category
        if not self.categories:
            raise ValueError("graph requires at least one category")

    def artifact(
        self,
        artifact_id: str,
        *,
        data: bytes | bytearray | memoryview | str | os.PathLike[str] | None = None,
        path: str | os.PathLike[str] | None = None,
        name: str | None = None,
        media_type: str = "application/octet-stream",
    ) -> Artifact:
        if artifact_id in self.artifacts:
            raise ValueError(f"duplicate artifact id: {artifact_id}")
        artifact = Artifact(artifact_id, data=data, path=path, name=name, media_type=media_type)
        self.artifacts[artifact_id] = artifact
        return artifact

    def node(
        self,
        node_id: str,
        *,
        label: str | None = None,
        kind: str = "entity",
        category: str = "metadata",
        spans: Iterable[Span] = (),
        properties: Mapping[str, Any] | None = None,
        fields: Mapping[str, Field | Any] | None = None,
        addresses: Mapping[str, Address | int] | None = None,
        parent: Node | str | None = None,
    ) -> Node:
        node_id = _identifier(node_id, "node id")
        if node_id in self.nodes:
            raise ValueError(f"duplicate node id: {node_id}")
        label = _identifier(label if label is not None else node_id, "node label")
        kind = _identifier(kind, "node kind")
        if category not in self.categories:
            raise ValueError(f"unknown semantic category: {category}")
        field_values = {
            _identifier(name, "field name"): value if isinstance(value, Field) else Field(value)
            for name, value in (fields or {}).items()
        }
        address_values: dict[str, Address] = {}
        for name, value in (addresses or {}).items():
            name = _identifier(name, "address name")
            address_values[name] = value if isinstance(value, Address) else Address(name, value)
        node = Node(
            node_id,
            label,
            kind,
            category,
            tuple(spans),
            dict(_json_value(dict(properties or {}), "node properties")),
            field_values,
            address_values,
        )
        self.nodes[node_id] = node
        if parent is not None:
            self.contain(parent, node)
        return node

    @staticmethod
    def _node_id(value: Node | str) -> str:
        return value.id if isinstance(value, Node) else _identifier(value, "node reference")

    def contain(self, parent: Node | str, child: Node | str) -> None:
        parent_id = self._node_id(parent)
        child_id = self._node_id(child)
        if parent_id not in self.nodes or child_id not in self.nodes:
            raise ValueError("containment references an unknown node")
        if parent_id == child_id:
            raise ValueError("a node cannot contain itself")
        if child_id in self._parents:
            raise ValueError(f"node already has a parent: {child_id}")
        cursor = parent_id
        while cursor in self._parents:
            cursor = self._parents[cursor]
            if cursor == child_id:
                raise ValueError("containment would create a cycle")
        self._parents[child_id] = parent_id

    def link(
        self,
        source: Node | str,
        target: Node | str,
        kind: str,
        *,
        properties: Mapping[str, Any] | None = None,
    ) -> Link:
        source_id = self._node_id(source)
        target_id = self._node_id(target)
        if source_id not in self.nodes or target_id not in self.nodes:
            raise ValueError("link references an unknown node")
        link = Link(
            source_id,
            target_id,
            _identifier(kind, "link kind"),
            dict(_json_value(dict(properties or {}), "link properties")),
        )
        self.links.append(link)
        return link

    def validate(self) -> None:
        artifact_objects = set(self.artifacts.values())
        for node in self.nodes.values():
            for span in (*node.spans, *(span for value in node.fields.values() for span in value.spans)):
                if span.artifact not in artifact_objects:
                    raise ValueError(f"node {node.id!r} references an artifact from another graph")
                if span.end > span.artifact.size:
                    raise ValueError(
                        f"span [{span.start}, {span.end}) for node {node.id!r} exceeds "
                        f"artifact {span.artifact.id!r} size {span.artifact.size}"
                    )
        for child, parent in self._parents.items():
            if child not in self.nodes or parent not in self.nodes:
                raise ValueError("containment references an unknown node")
        for link in self.links:
            if link.source not in self.nodes or link.target not in self.nodes:
                raise ValueError("link references an unknown node")

    def to_dict(self, *, include_data: bool = False) -> dict[str, Any]:
        self.validate()
        nodes = []
        for node in self.nodes.values():
            value: dict[str, Any] = {
                "id": node.id,
                "label": node.label,
                "kind": node.kind,
                "category": node.category,
                "spans": [span.to_dict() for span in node.spans],
                "properties": dict(node.properties),
                "fields": {name: field_value.to_dict() for name, field_value in node.fields.items()},
                "addresses": {
                    name: {"space": address.space, "value": address.value} for name, address in node.addresses.items()
                },
            }
            if node.id in self._parents:
                value["parent"] = self._parents[node.id]
            nodes.append(value)
        return {
            "format": FORMAT,
            "version": FORMAT_VERSION,
            "title": self.title,
            "categories": [
                {"id": category.id, "label": category.label, "color": category.color}
                for category in self.categories.values()
            ],
            "artifacts": [artifact.to_dict(include_data=include_data) for artifact in self.artifacts.values()],
            "nodes": nodes,
            "links": [
                {
                    "source": link.source,
                    "target": link.target,
                    "kind": link.kind,
                    "properties": dict(link.properties),
                }
                for link in self.links
            ],
        }

    def dumps(self, *, include_data: bool = False, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(include_data=include_data), ensure_ascii=False, indent=indent, allow_nan=False)

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        artifact_data: Mapping[str, bytes | bytearray | memoryview | str | os.PathLike[str]] | None = None,
    ) -> ArtifactGraph:
        if value.get("format") != FORMAT or value.get("version") != FORMAT_VERSION:
            raise ValueError("unsupported artifact graph serialization format or version")
        categories = tuple(Category(item["id"], item["label"], item["color"]) for item in value["categories"])
        graph = cls(value["title"], categories=categories)
        artifact_data = artifact_data or {}
        for item in value["artifacts"]:
            source: bytes | bytearray | memoryview | str | os.PathLike[str]
            if "data_base64" in item:
                try:
                    source = base64.b64decode(item["data_base64"], validate=True)
                except (ValueError, TypeError) as error:
                    raise ValueError(f"invalid base64 data for artifact {item.get('id')!r}") from error
            elif item["id"] in artifact_data:
                source = artifact_data[item["id"]]
            else:
                raise ValueError(f"serialized artifact {item['id']!r} has no bytes; supply artifact_data")
            artifact = graph.artifact(
                item["id"],
                data=source,
                name=item["name"],
                media_type=item.get("media_type", "application/octet-stream"),
            )
            if artifact.size != item["size"] or artifact.sha256 != item["sha256"]:
                raise ValueError(f"serialized artifact {item['id']!r} does not match its bytes")
        for item in value["nodes"]:
            spans = tuple(
                Span(graph.artifacts[span["artifact"]], span["start"], span["end"]) for span in item.get("spans", ())
            )
            fields: dict[str, Field] = {}
            for name, field_value in item.get("fields", {}).items():
                field_spans = tuple(
                    Span(graph.artifacts[span["artifact"]], span["start"], span["end"])
                    for span in field_value.get("spans", ())
                )
                fields[name] = Field(field_value.get("value"), field_spans, field_value.get("display"))
            addresses = {
                name: Address(address["space"], address["value"]) for name, address in item.get("addresses", {}).items()
            }
            graph.node(
                item["id"],
                label=item["label"],
                kind=item["kind"],
                category=item["category"],
                spans=spans,
                properties=item.get("properties", {}),
                fields=fields,
                addresses=addresses,
            )
        for item in value["nodes"]:
            if parent := item.get("parent"):
                graph.contain(parent, item["id"])
        for item in value.get("links", ()):
            graph.link(item["source"], item["target"], item["kind"], properties=item.get("properties", {}))
        graph.validate()
        return graph

    @classmethod
    def loads(
        cls,
        value: str,
        *,
        artifact_data: Mapping[str, bytes | bytearray | memoryview | str | os.PathLike[str]] | None = None,
    ) -> ArtifactGraph:
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("artifact graph serialization must be a JSON object")
        return cls.from_dict(parsed, artifact_data=artifact_data)
