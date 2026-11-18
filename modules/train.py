"""Train an Isolation Forest using healthy recordings."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .anomaly_detection import (
    DEFAULT_MODEL_PATH,
    create_pipeline,
    save_model,
)
from .feature_extraction import FEATURE_NAMES, FeatureExtractionError, extract_features

logger = logging.getLogger(__name__)
WAV_SUFFIXES = {".wav"}


def train_model(
    dataset_path: str | Path,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    contamination: float | str = "auto",
    random_state: int = 42,
) -> Pipeline:
    """Fit and save a model from healthy WAV files in the dataset.

    ``dataset/normal`` is the preferred location. ``dataset/healthy`` is
    accepted as a compatibility fallback for the existing dashboard.
    """
    dataset = Path(dataset_path)
    normal_dir = dataset / "normal"
    healthy_dir = dataset / "healthy"
    normal_has_wavs = normal_dir.is_dir() and any(
        path.is_file() and path.suffix.lower() in WAV_SUFFIXES
        for path in normal_dir.rglob("*")
    )
    healthy_has_wavs = healthy_dir.is_dir() and any(
        path.is_file() and path.suffix.lower() in WAV_SUFFIXES
        for path in healthy_dir.rglob("*")
    )
    if not normal_has_wavs and healthy_has_wavs:
        normal_dir = healthy_dir
    elif not normal_dir.is_dir() and healthy_dir.is_dir():
        normal_dir = healthy_dir
    if not normal_dir.is_dir():
        raise FileNotFoundError(
            f"Healthy recordings directory not found under {dataset}. "
            "Add WAV recordings to dataset/normal/ before training."
        )

    audio_paths = sorted(
        path
        for path in normal_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in WAV_SUFFIXES
    )
    if not audio_paths:
        raise ValueError(
            f"No WAV recordings found in {normal_dir}. "
            f"Add healthy machine WAV recordings to {normal_dir}."
        )

    feature_rows: list[list[float]] = []
    for audio_path in audio_paths:
        try:
            features = extract_features(audio_path)
        except (FeatureExtractionError, OSError) as exc:
            logger.warning("Skipping unusable training recording %s: %s", audio_path, exc)
            continue
        feature_rows.append([features[name] for name in FEATURE_NAMES])

    if len(feature_rows) < 2:
        raise ValueError(
            f"Only {len(feature_rows)} valid healthy recording(s) were found in "
            f"{normal_dir}; at least two are required to train."
        )

    matrix = np.asarray(feature_rows, dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("Training features contain non-finite values.")
    model = create_pipeline(contamination=contamination, random_state=random_state)
    model.fit(pd.DataFrame(matrix, columns=FEATURE_NAMES))
    save_model(model, model_path)
    logger.info("Trained using %d healthy recordings", len(feature_rows))
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--contamination", type=float, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    contamination = args.contamination if args.contamination is not None else "auto"
    model = train_model(
        args.dataset,
        model_path=args.model,
        contamination=contamination,
        random_state=args.random_state,
    )
    count = model.named_steps["scaler"].n_samples_seen_
    print(f"Trained model on {count} healthy recordings; saved to {args.model}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
