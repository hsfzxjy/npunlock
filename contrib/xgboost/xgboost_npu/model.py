from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


def _f32(value: object) -> float:
    result = float(np.float32(value))
    if not math.isfinite(result):
        raise ValueError("model contains a non-finite threshold or leaf value")
    return result


def _base_score(value: object) -> float:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            decoded = json.loads(stripped)
            if not isinstance(decoded, list) or len(decoded) != 1:
                raise ValueError("binary model base_score must contain one value")
            value = decoded[0]
    probability = _f32(value)
    if not 0.0 < probability < 1.0:
        raise ValueError("binary:logistic base_score must be strictly between zero and one")
    return float(np.float32(math.log(probability / (1.0 - probability))))


def _integer_list(value: object, name: str, count: int) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) != count:
        raise ValueError(f"tree {name} must contain {count} entries")
    result = tuple(int(item) for item in value)
    if any(isinstance(item, bool) or int(item) != item for item in value):
        raise ValueError(f"tree {name} must contain integers")
    return result


@dataclass(frozen=True, slots=True)
class Tree:
    left: tuple[int, ...]
    right: tuple[int, ...]
    split_feature: tuple[int, ...]
    split_or_leaf: tuple[float, ...]
    default_left: tuple[bool, ...]

    @property
    def node_count(self) -> int:
        return len(self.left)

    @property
    def max_depth(self) -> int:
        maximum = 0
        seen: set[int] = set()
        stack = [(0, 0)]
        while stack:
            node, depth = stack.pop()
            if node in seen:
                raise ValueError("tree contains a cycle or a multiply referenced node")
            seen.add(node)
            maximum = max(maximum, depth)
            if self.left[node] != -1:
                stack.append((self.left[node], depth + 1))
                stack.append((self.right[node], depth + 1))
        if len(seen) != self.node_count:
            raise ValueError("tree contains unreachable nodes")
        return maximum

    def validate(self, feature_count: int) -> None:
        count = self.node_count
        if count == 0 or not (
            len(self.right)
            == len(self.split_feature)
            == len(self.split_or_leaf)
            == len(self.default_left)
            == count
        ):
            raise ValueError("tree arrays must be nonempty and equal length")
        parent_counts = [0] * count
        for index, (left, right) in enumerate(zip(self.left, self.right)):
            is_leaf = left == -1 and right == -1
            if (left == -1) != (right == -1):
                raise ValueError(f"tree node {index} has only one child")
            if is_leaf:
                continue
            if not 0 <= self.split_feature[index] < feature_count:
                raise ValueError(f"tree node {index} has an out-of-range feature index")
            if not (0 <= left < count and 0 <= right < count) or left == right:
                raise ValueError(f"tree node {index} has invalid child indices")
            parent_counts[left] += 1
            parent_counts[right] += 1
        if parent_counts[0] != 0 or any(value != 1 for value in parent_counts[1:]):
            raise ValueError("tree root/child parent counts are invalid")
        self.max_depth

    def predict_one(self, features: np.ndarray) -> np.float32:
        node = 0
        while self.left[node] != -1:
            value = np.float32(features[self.split_feature[node]])
            if np.isnan(value):
                node = self.left[node] if self.default_left[node] else self.right[node]
            else:
                node = self.left[node] if value < np.float32(self.split_or_leaf[node]) else self.right[node]
        return np.float32(self.split_or_leaf[node])


@dataclass(frozen=True, slots=True)
class Ensemble:
    feature_count: int
    base_margin: float
    trees: tuple[Tree, ...]
    feature_names: tuple[str, ...]
    xgboost_version: tuple[int, ...]

    @property
    def max_depth(self) -> int:
        return max(tree.max_depth for tree in self.trees)

    @property
    def node_count(self) -> int:
        return sum(tree.node_count for tree in self.trees)

    def predict_margin(self, features: object) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.feature_count:
            raise ValueError(f"features must have shape (N, {self.feature_count})")
        result = np.empty(values.shape[0], dtype=np.float32)
        for row_index, row in enumerate(values):
            margin = np.float32(self.base_margin)
            for tree in self.trees:
                margin = np.float32(margin + tree.predict_one(row))
            result[row_index] = margin
        return result

    def predict_probability(self, features: object) -> np.ndarray:
        margin = self.predict_margin(features)
        result = np.empty_like(margin)
        positive = margin >= 0.0
        result[positive] = np.float32(1.0) / (np.float32(1.0) + np.exp(-margin[positive], dtype=np.float32))
        exponential = np.exp(margin[~positive], dtype=np.float32)
        result[~positive] = exponential / (np.float32(1.0) + exponential)
        return result


def _parse_tree(raw: Mapping[str, object], feature_count: int) -> Tree:
    parameters = raw.get("tree_param")
    if not isinstance(parameters, Mapping):
        raise ValueError("tree_param is missing")
    node_count = int(parameters.get("num_nodes", 0))
    if node_count <= 0 or int(parameters.get("num_deleted", 0)) != 0:
        raise ValueError("deleted or empty trees are unsupported")
    if int(parameters.get("num_feature", 0)) != feature_count:
        raise ValueError("tree num_feature does not match the learner")
    split_type = _integer_list(raw.get("split_type"), "split_type", node_count)
    if any(split_type):
        raise ValueError("categorical XGBoost splits are unsupported")
    for name in ("categories", "categories_nodes", "categories_segments", "categories_sizes"):
        if raw.get(name, []) != []:
            raise ValueError("categorical XGBoost metadata is unsupported")
    values = raw.get("split_conditions")
    if not isinstance(values, list) or len(values) != node_count:
        raise ValueError(f"tree split_conditions must contain {node_count} entries")
    default_left = _integer_list(raw.get("default_left"), "default_left", node_count)
    if any(value not in (0, 1) for value in default_left):
        raise ValueError("tree default_left entries must be zero or one")
    tree = Tree(
        left=_integer_list(raw.get("left_children"), "left_children", node_count),
        right=_integer_list(raw.get("right_children"), "right_children", node_count),
        split_feature=_integer_list(raw.get("split_indices"), "split_indices", node_count),
        split_or_leaf=tuple(_f32(value) for value in values),
        default_left=tuple(bool(value) for value in default_left),
    )
    tree.validate(feature_count)
    return tree


def load_ensemble(source: str | Path | bytes | bytearray | Mapping[str, object]) -> Ensemble:
    if isinstance(source, Mapping):
        root = source
    else:
        payload = bytes(source) if isinstance(source, (bytes, bytearray)) else Path(source).read_bytes()
        decoded = json.loads(payload)
        if not isinstance(decoded, Mapping):
            raise ValueError("XGBoost model root must be an object")
        root = decoded
    learner = root.get("learner")
    if not isinstance(learner, Mapping):
        raise ValueError("XGBoost learner object is missing")
    objective = learner.get("objective")
    if not isinstance(objective, Mapping) or objective.get("name") != "binary:logistic":
        raise ValueError("only the binary:logistic objective is supported")
    parameters = learner.get("learner_model_param")
    if not isinstance(parameters, Mapping):
        raise ValueError("learner_model_param is missing")
    feature_count = int(parameters.get("num_feature", 0))
    if feature_count <= 0 or int(parameters.get("num_target", 1)) != 1 or int(parameters.get("num_class", 0)) != 0:
        raise ValueError("only one-target binary models with numerical features are supported")
    booster = learner.get("gradient_booster")
    if not isinstance(booster, Mapping) or booster.get("name") != "gbtree":
        raise ValueError("only gbtree boosters are supported")
    model = booster.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("gradient_booster model is missing")
    raw_trees = model.get("trees")
    if not isinstance(raw_trees, list) or not raw_trees:
        raise ValueError("model must contain at least one tree")
    tree_info = model.get("tree_info")
    if not isinstance(tree_info, list) or len(tree_info) != len(raw_trees) or any(int(value) != 0 for value in tree_info):
        raise ValueError("only one output group is supported")
    trees = tuple(_parse_tree(tree, feature_count) for tree in raw_trees if isinstance(tree, Mapping))
    if len(trees) != len(raw_trees):
        raise ValueError("every tree must be an object")
    names = learner.get("feature_names", [])
    if not isinstance(names, list) or (names and len(names) != feature_count):
        raise ValueError("feature_names must be empty or match num_feature")
    version = root.get("version", [])
    if not isinstance(version, list) or any(not isinstance(value, int) for value in version):
        raise ValueError("XGBoost version must be an integer array")
    return Ensemble(
        feature_count=feature_count,
        base_margin=_base_score(parameters.get("base_score", "0.5")),
        trees=trees,
        feature_names=tuple(str(value) for value in names),
        xgboost_version=tuple(version),
    )
