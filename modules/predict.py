"""Audio-file prediction API and command-line interface."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from sklearn.pipeline import Pipeline

from .anomaly_detection import DEFAULT_MODEL_PATH, load_model, predict_from_features
from .feature_extraction import extract_features

logger = logging.getLogger(__name__)


def predict_anomaly(
    audio_path: str | Path, model: Pipeline | None = None
) -> dict[str, float | str | bool]:
    """Extract features from an audio file and classify it."""
    fitted_model = model if model is not None else load_model()
    return predict_from_features(extract_features(audio_path), fitted_model)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_path", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()
    result = predict_anomaly(args.audio_path, load_model(args.model))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
