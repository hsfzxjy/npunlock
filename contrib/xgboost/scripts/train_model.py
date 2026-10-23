from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


SEED = 1729


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and export the reproducible XGBoost demonstration model")
    parser.add_argument("--output-dir", type=Path, default=Path("contrib/xgboost/artifacts/model"))
    parser.add_argument("--trees", type=int, default=32)
    parser.add_argument("--max-depth", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = options()
    if args.trees <= 0 or args.max_depth <= 0:
        raise SystemExit("--trees and --max-depth must be positive")
    dataset = load_breast_cancer()
    features = np.asarray(dataset.data, dtype=np.float32)
    labels = np.asarray(dataset.target, dtype=np.int32)
    indices = np.arange(features.shape[0], dtype=np.int32)
    train_indices, validation_indices = train_test_split(
        indices,
        test_size=0.25,
        random_state=SEED,
        stratify=labels,
    )
    parameters: dict[str, object] = {
        "n_estimators": args.trees,
        "max_depth": args.max_depth,
        "learning_rate": 0.15,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "min_child_weight": 1.0,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": SEED,
        "n_jobs": 1,
    }
    classifier = XGBClassifier(**parameters)
    classifier.fit(features[train_indices], labels[train_indices])
    validation_features = features[validation_indices]
    validation_labels = labels[validation_indices]
    probabilities = classifier.predict_proba(validation_features)[:, 1]
    predictions = (probabilities >= 0.5).astype(np.int32)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "breast_cancer_xgboost.json"
    onnx_path = output_dir / "breast_cancer_xgboost.onnx"
    validation_path = output_dir / "validation.npz"
    metadata_path = output_dir / "metadata.json"
    classifier.get_booster().save_model(model_path)
    onnx_model = convert_xgboost(
        classifier,
        name="breast_cancer_xgboost",
        initial_types=[("features", FloatTensorType([None, features.shape[1]]))],
    )
    onnx_path.write_bytes(onnx_model.SerializeToString())
    np.savez_compressed(
        validation_path,
        features=validation_features,
        labels=validation_labels,
        source_indices=validation_indices,
    )
    metadata = {
        "schema": "npunlock-xgboost-model-v1",
        "dataset": {
            "name": "Wisconsin Diagnostic Breast Cancer",
            "source": "scikit-learn load_breast_cancer",
            "samples": int(features.shape[0]),
            "features": int(features.shape[1]),
            "classes": [str(value) for value in dataset.target_names],
            "feature_names": [str(value) for value in dataset.feature_names],
            "validation_samples": int(validation_features.shape[0]),
            "split_seed": SEED,
        },
        "training": parameters,
        "validation": {
            "accuracy": float(accuracy_score(validation_labels, predictions)),
            "roc_auc": float(roc_auc_score(validation_labels, probabilities)),
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scikit-learn", "xgboost", "onnx", "onnxmltools")
        },
        "artifacts": {
            "model": {"path": model_path.name, "sha256": sha256(model_path)},
            "onnx": {"path": onnx_path.name, "sha256": sha256(onnx_path)},
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
