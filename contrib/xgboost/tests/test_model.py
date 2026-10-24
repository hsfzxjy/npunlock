from __future__ import annotations

import unittest

import numpy as np

from xgboost_npu.generate import generate_kernel_source
from xgboost_npu.model import load_ensemble
from xgboost_npu.packing import FP16_TRANSPORT, FP32_TRANSPORT, pack_features
from xgboost_npu.runtime import (
    RUNTIME_CHUNK_ELEMENTS,
    RUNTIME_REPLICAS,
    pack_runtime_model,
    pack_runtime_features,
    runtime_batch_capacity,
)


def stump_model() -> dict[str, object]:
    tree = {
        "id": 0,
        "tree_param": {
            "num_deleted": "0",
            "num_feature": "2",
            "num_nodes": "3",
            "size_leaf_vector": "1",
        },
        "left_children": [1, -1, -1],
        "right_children": [2, -1, -1],
        "split_indices": [0, 0, 0],
        "split_conditions": [1.0, -2.0, 3.0],
        "default_left": [1, 0, 0],
        "split_type": [0, 0, 0],
        "categories": [],
        "categories_nodes": [],
        "categories_segments": [],
        "categories_sizes": [],
    }
    return {
        "version": [3, 4, 1],
        "learner": {
            "feature_names": ["a", "b"],
            "feature_types": ["float", "float"],
            "learner_model_param": {
                "base_score": "[5E-1]",
                "num_class": "0",
                "num_feature": "2",
                "num_target": "1",
            },
            "objective": {"name": "binary:logistic"},
            "gradient_booster": {
                "name": "gbtree",
                "model": {
                    "tree_info": [0],
                    "trees": [tree],
                    "iteration_indptr": [0, 1],
                    "gbtree_model_param": {
                        "num_trees": "1",
                        "num_parallel_tree": "1",
                    },
                },
            },
        },
    }


class ModelTests(unittest.TestCase):
    def test_threshold_and_missing_value_routing(self) -> None:
        ensemble = load_ensemble(stump_model())
        features = np.asarray([[0.5, 0.0], [1.0, 0.0], [np.nan, 0.0]], dtype=np.float32)
        np.testing.assert_array_equal(
            ensemble.predict_margin(features),
            np.asarray([-2.0, 3.0, -2.0], dtype=np.float32),
        )

    def test_generator_contains_the_model_branch(self) -> None:
        ensemble = load_ensemble(stump_model())
        source = generate_kernel_source(ensemble).decode("utf-8")
        self.assertIn("value != value", source)
        self.assertIn("value < 1.0f", source)
        self.assertIn("expf", source)
        packed_model = pack_runtime_model(ensemble).reshape(-1)
        root_metadata = int(packed_model.view(np.uint32)[5])
        self.assertEqual(root_metadata & ~0x7FFFF, 0x3F800000)

    def test_feature_payload_is_replicated(self) -> None:
        features = np.asarray([[1.0, np.nan], [-2.5, 4.0]], dtype=np.float32)
        for transport in (FP16_TRANSPORT, FP32_TRANSPORT):
            with self.subTest(transport=transport.name):
                packed = pack_features(
                    features,
                    feature_count=2,
                    transport=transport,
                ).reshape(-1)
                first = packed[: transport.chunk_elements]
                second = packed[transport.chunk_elements :]
                np.testing.assert_array_equal(
                    first.view(f"V{transport.numpy_dtype.itemsize}"),
                    second.view(f"V{transport.numpy_dtype.itemsize}"),
                )

    def test_runtime_features_balance_across_four_invocations(self) -> None:
        self.assertEqual(runtime_batch_capacity(30), 136)
        features = np.arange(8 * 30, dtype=np.float32).reshape(8, 30)
        chunks = pack_runtime_features(features, feature_count=30).reshape(
            RUNTIME_REPLICAS,
            RUNTIME_CHUNK_ELEMENTS,
        )
        self.assertEqual(tuple(int(chunk[0]) for chunk in chunks), (2, 2, 2, 2))
        for replica in range(RUNTIME_REPLICAS):
            start = replica * 2
            np.testing.assert_array_equal(
                chunks[replica, 4 : 4 + 2 * 30],
                features[start : start + 2].reshape(-1),
            )


if __name__ == "__main__":
    unittest.main()
