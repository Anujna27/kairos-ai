"""Evaluate anomaly predictions against normal/abnormal directory labels."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from .anomaly_detection import DEFAULT_MODEL_PATH, load_model
from .feature_extraction import FeatureExtractionError
from .predict import predict_anomaly

logger = logging.getLogger(__name__)


def evaluate_model(
    model: Pipeline,
    normal_dir: str | Path,
    abnormal_dir: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate labeled WAV files; abnormal is the positive class.

    ROC-AUC uses the uncalibrated anomaly score (negative decision function),
    not a probability. When ``output_path`` is supplied, JSON results and a
    confusion-matrix PNG are written next to it.
    """
    paths_and_labels = [
        (Path(normal_dir), 0),
        (Path(abnormal_dir), 1),
    ]
    y_true: list[int] = []
    y_pred: list[int] = []
    anomaly_scores: list[float] = []
    evaluated_files = 0

    for folder, label in paths_and_labels:
        if not folder.is_dir():
            logger.warning("Evaluation directory does not exist: %s", folder)
            continue
        for audio_path in sorted(folder.rglob("*")):
            if not audio_path.is_file() or audio_path.suffix.lower() != ".wav":
                continue
            try:
                result = predict_anomaly(audio_path, model)
            except (FeatureExtractionError, OSError) as exc:
                logger.warning("Skipping unusable evaluation recording %s: %s", audio_path, exc)
                continue
            y_true.append(label)
            y_pred.append(1 if result["is_anomaly"] else 0)
            anomaly_scores.append(float(result["anomaly_score"]))
            evaluated_files += 1

    if not evaluated_files:
        raise ValueError(
            "No valid WAV recordings were available for evaluation. "
            "Add labeled recordings under dataset/normal/ and dataset/abnormal/."
        )

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    true_normal, false_positive, false_negative, true_abnormal = matrix.ravel()
    results: dict[str, Any] = {
        "evaluated_recordings": evaluated_files,
        "labels": {"normal": 0, "abnormal": 1},
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": ["NORMAL", "ANOMALY"],
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": (
            float(false_positive / (false_positive + true_normal))
            if false_positive + true_normal
            else None
        ),
        "false_negative_rate": (
            float(false_negative / (false_negative + true_abnormal))
            if false_negative + true_abnormal
            else None
        ),
        "roc_auc": (
            float(roc_auc_score(y_true, anomaly_scores))
            if len(set(y_true)) == 2
            else None
        ),
        "score_note": "Anomaly scores are uncalibrated; they are not probabilities.",
    }

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        plot_path = output.with_name(f"{output.stem}_confusion_matrix.png")
        figure, axis = plt.subplots()
        image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
        figure.colorbar(image, ax=axis)
        axis.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["NORMAL", "ANOMALY"],
            yticklabels=["NORMAL", "ANOMALY"],
            xlabel="Predicted label",
            ylabel="True label",
        )
        threshold = matrix.max() / 2
        for row in range(2):
            for column in range(2):
                axis.text(
                    column,
                    row,
                    str(matrix[row, column]),
                    ha="center",
                    va="center",
                    color="white" if matrix[row, column] > threshold else "black",
                )
        figure.tight_layout()
        figure.savefig(plot_path)
        plt.close(figure)
        results["confusion_matrix_plot"] = str(plot_path)
        output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        logger.info("Saved evaluation results to %s", output)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=Path("evaluation_results.json"))
    args = parser.parse_args()
    model = load_model(args.model)
    results = evaluate_model(
        model,
        args.dataset / "normal",
        args.dataset / "abnormal",
        args.output,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
