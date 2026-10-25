from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files
from typing import Any, Mapping

from .model import Artifact, Field, Span


SCHEMA_FORMAT = "npunlock.viz-schema"
SCHEMA_VERSION = 1

_ENCODING_SIZES = {
    "u8": 1,
    "u16le": 2,
    "u32le": 4,
    "u64le": 8,
    "i64le": 8,
    "u64le-hi32": 8,
    "u64le-lo32": 8,
}
_DISPLAYS = {None, "hex", "hex-nonnegative"}
_CONFIDENCE = {"confirmed", "strong-inference", "execution-tested-inference", "raw"}
_ROOT_KEYS = {"format", "version", "id", "nodes", "records", "lookups", "rules"}


def _text(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{what} must be a non-empty string without surrounding whitespace")
    return value


def validate_schema(value: Any) -> dict[str, Any]:
    """Validate the deliberately small, non-executable visualization descriptor format."""

    if not isinstance(value, dict):
        raise ValueError("visualization schema must be a JSON object")
    unknown = set(value) - _ROOT_KEYS
    if unknown:
        raise ValueError(f"visualization schema has unknown top-level keys: {sorted(unknown)!r}")
    if value.get("format") != SCHEMA_FORMAT or value.get("version") != SCHEMA_VERSION:
        raise ValueError("unsupported visualization schema format or version")
    _text(value.get("id"), "schema id")
    nodes = value.get("nodes")
    records = value.get("records")
    if not isinstance(nodes, dict) or not isinstance(records, dict):
        raise ValueError("visualization schema nodes and records must be objects")
    for node_id, node in nodes.items():
        _text(node_id, "node descriptor id")
        if not isinstance(node, dict) or set(node) != {"kind", "category", "description"}:
            raise ValueError(f"node descriptor {node_id!r} must contain kind, category, and description")
        for key in ("kind", "category", "description"):
            _text(node[key], f"node descriptor {node_id!r} {key}")
    for record_id, record in records.items():
        _text(record_id, "record descriptor id")
        if not isinstance(record, dict) or set(record) != {"size", "fields"}:
            raise ValueError(f"record descriptor {record_id!r} must contain size and fields")
        record_size = record["size"]
        if isinstance(record_size, bool) or not isinstance(record_size, int) or record_size <= 0:
            raise ValueError(f"record descriptor {record_id!r} size must be a positive integer")
        fields_value = record["fields"]
        if not isinstance(fields_value, list) or not fields_value:
            raise ValueError(f"record descriptor {record_id!r} fields must be a non-empty array")
        field_ids: set[str] = set()
        for field in fields_value:
            if not isinstance(field, dict):
                raise ValueError(f"record descriptor {record_id!r} field must be an object")
            field_id = _text(field.get("id"), f"record descriptor {record_id!r} field id")
            if field_id in field_ids:
                raise ValueError(f"record descriptor {record_id!r} has duplicate field {field_id!r}")
            field_ids.add(field_id)
            allowed = {
                "id",
                "computed",
                "offset",
                "size",
                "encoding",
                "display",
                "description",
                "value_map",
                "confidence",
                "scope",
            }
            extra = set(field) - allowed
            if extra:
                raise ValueError(f"field {record_id}.{field_id} has unknown keys: {sorted(extra)!r}")
            _text(field.get("description"), f"field {record_id}.{field_id} description")
            confidence = field.get("confidence")
            if confidence is not None and confidence not in _CONFIDENCE:
                raise ValueError(f"field {record_id}.{field_id} has unsupported confidence {confidence!r}")
            if "scope" in field:
                _text(field["scope"], f"field {record_id}.{field_id} scope")
            computed = field.get("computed", False)
            if not isinstance(computed, bool):
                raise ValueError(f"field {record_id}.{field_id} computed must be boolean")
            if computed:
                forbidden = {"offset", "size", "encoding", "display", "value_map"} & set(field)
                if forbidden:
                    raise ValueError(f"computed field {record_id}.{field_id} cannot contain {sorted(forbidden)!r}")
                continue
            required = {"offset", "size", "encoding"}
            if not required.issubset(field):
                raise ValueError(f"field {record_id}.{field_id} requires offset, size, and encoding")
            offset, size, encoding = field["offset"], field["size"], field["encoding"]
            if any(isinstance(item, bool) or not isinstance(item, int) for item in (offset, size)):
                raise ValueError(f"field {record_id}.{field_id} offset and size must be integers")
            if offset < 0 or size <= 0 or offset > record_size or size > record_size - offset:
                raise ValueError(f"field {record_id}.{field_id} is outside its record")
            if encoding not in _ENCODING_SIZES or _ENCODING_SIZES[encoding] != size:
                raise ValueError(f"field {record_id}.{field_id} encoding does not match its size")
            if field.get("display") not in _DISPLAYS:
                raise ValueError(f"field {record_id}.{field_id} has unsupported display")
            value_map = field.get("value_map")
            if value_map is not None and (
                not isinstance(value_map, dict) or any(not isinstance(key, str) for key in value_map)
            ):
                raise ValueError(f"field {record_id}.{field_id} value_map must be an object with string keys")
    lookups = value.get("lookups", {})
    rules = value.get("rules", {})
    if not isinstance(lookups, dict) or not isinstance(rules, dict):
        raise ValueError("visualization schema lookups and rules must be objects")
    json.dumps(lookups, ensure_ascii=False, allow_nan=False)
    for rule_id, rule_values in rules.items():
        _text(rule_id, "rule id")
        if not isinstance(rule_values, list):
            raise ValueError(f"schema rule {rule_id!r} must be an array")
        for rule in rule_values:
            if not isinstance(rule, dict) or "value" not in rule:
                raise ValueError(f"schema rule {rule_id!r} entries must be objects with a value")
            if set(rule) - {"value", "equals", "prefixes", "contains"}:
                raise ValueError(f"schema rule {rule_id!r} has unknown matcher keys")
            _text(rule["value"], f"schema rule {rule_id!r} value")
            matchers = 0
            for key in ("equals", "prefixes", "contains"):
                if key not in rule:
                    continue
                matchers += 1
                if not isinstance(rule[key], list) or not rule[key]:
                    raise ValueError(f"schema rule {rule_id!r} {key} must be a non-empty array")
                for item in rule[key]:
                    _text(item, f"schema rule {rule_id!r} {key} item")
            if matchers != 1:
                raise ValueError(f"schema rule {rule_id!r} entry must use exactly one matcher")
    return value


@lru_cache(maxsize=None)
def load_schema(name: str) -> dict[str, Any]:
    if name not in {"elf.json", "npu3720-act-compiler-8.3.json"}:
        raise ValueError(f"unknown bundled visualization schema: {name}")
    resource = files("npunlock.viz").joinpath("schemas", name)
    try:
        value = json.loads(resource.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load bundled visualization schema {name!r}") from error
    return validate_schema(value)


def node_metadata(schema: Mapping[str, Any], node_id: str) -> dict[str, str]:
    try:
        node = schema["nodes"][node_id]
        return {"kind": node["kind"], "category": node["category"], "description": node["description"]}
    except (KeyError, TypeError) as error:
        raise ValueError(f"schema has no node descriptor {node_id!r}") from error


def lookup(schema: Mapping[str, Any], lookup_id: str, key: str, default: Any = None) -> Any:
    values = schema.get("lookups", {}).get(lookup_id, {})
    if not isinstance(values, dict):
        raise ValueError(f"schema lookup {lookup_id!r} is not an object")
    return values.get(key, default)


def match_rule(schema: Mapping[str, Any], rule_id: str, text: str, default: str) -> str:
    lowered = text.lower()
    for rule in schema.get("rules", {}).get(rule_id, []):
        if "equals" in rule and lowered in rule["equals"]:
            return rule["value"]
        if "prefixes" in rule and any(lowered.startswith(value) for value in rule["prefixes"]):
            return rule["value"]
        if "contains" in rule and any(value in lowered for value in rule["contains"]):
            return rule["value"]
    return default


def decode_record_fields(
    schema: Mapping[str, Any],
    record_id: str,
    artifact: Artifact,
    data: bytes,
    base: int,
    *,
    computed: Mapping[str, Any] | None = None,
) -> dict[str, Field]:
    try:
        record = schema["records"][record_id]
    except (KeyError, TypeError) as error:
        raise ValueError(f"schema has no record descriptor {record_id!r}") from error
    record_size = record["size"]
    if isinstance(base, bool) or not isinstance(base, int) or base < 0:
        raise ValueError("record base must be a non-negative integer")
    if base > len(data) or record_size > len(data) - base or base + record_size > artifact.size:
        raise ValueError(f"{record_id} record is outside artifact {artifact.id!r}")
    computed = computed or {}
    result: dict[str, Field] = {}
    for descriptor in record["fields"]:
        field_id = descriptor["id"]
        if descriptor.get("computed", False):
            if field_id not in computed:
                raise ValueError(f"computed field {record_id}.{field_id} was not supplied")
            result[field_id] = Field(computed[field_id], description=descriptor["description"])
            continue
        start = base + descriptor["offset"]
        end = start + descriptor["size"]
        raw = data[start:end]
        encoding = descriptor["encoding"]
        if encoding == "i64le":
            value: Any = int.from_bytes(raw, "little", signed=True)
        else:
            value = int.from_bytes(raw, "little", signed=False)
            if encoding == "u64le-hi32":
                value >>= 32
            elif encoding == "u64le-lo32":
                value &= 0xFFFFFFFF
        value_map = descriptor.get("value_map")
        if value_map is not None:
            value = value_map.get(str(value), value)
        display_kind = descriptor.get("display")
        display = None
        if display_kind == "hex":
            display = hex(value)
        elif display_kind == "hex-nonnegative":
            display = hex(value) if value >= 0 else str(value)
        result[field_id] = Field(value, [Span(artifact, start, end)], display, descriptor["description"])
    extra = set(computed) - {field["id"] for field in record["fields"] if field.get("computed", False)}
    if extra:
        raise ValueError(f"unused computed fields for {record_id}: {sorted(extra)!r}")
    return result


__all__ = [
    "SCHEMA_FORMAT",
    "SCHEMA_VERSION",
    "decode_record_fields",
    "load_schema",
    "lookup",
    "match_rule",
    "node_metadata",
    "validate_schema",
]
