"""Isolation Forest model construction, persistence, and inference."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .feature_extraction import FEATURE_NAMES

logger = logging.getLogger(__name__)
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "isolation_forest.joblib"


def create_pipeline(
    contamination: float | str = "auto", random_state: int = 42
) -> Pipeline:
    """Create the scaled Isolation Forest pipeline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "isolation_forest",
                IsolationForest(
                    contamination=contamination,
                    random_state=random_state,
                ),
            ),
        ]
    )


def save_model(model: Pipeline, model_path: str | Path = DEFAULT_MODEL_PATH) -> Path:
    """Persist a complete fitted pipeline and return its destination."""
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Saved anomaly detection model to %s", path)
    return path


def load_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> Pipeline:
    """Load a persisted fitted pipeline."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Model not found at {path}. Train a model before making predictions."
        )
    model: Any = joblib.load(path)
    if not isinstance(model, Pipeline) or "isolation_forest" not in model.named_steps:
        raise ValueError(f"File does not contain an EchoSense Isolation Forest: {path}")
    return model


def _ordered_feature_vector(features: dict[str, float]) -> pd.DataFrame:
    """Validate and order a feature mapping according to the training schema."""
    missing = [name for name in FEATURE_NAMES if name not in features]
    if missing:
        raise ValueError(f"Missing required audio features: {', '.join(missing)}")
    try:
        vector = np.asarray([features[name] for name in FEATURE_NAMES], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Audio features must all be numeric") from exc
    if not np.isfinite(vector).all():
        raise ValueError("Audio features must all be finite")
    return pd.DataFrame([vector], columns=FEATURE_NAMES)


def predict_from_features(
    features: dict[str, float], model: Pipeline | None = None
) -> dict[str, float | str | bool]:
    """Classify a feature mapping and return uncalibrated Isolation Forest scores."""
    fitted_model = model if model is not None else load_model()
    vector = _ordered_feature_vector(features)
    prediction = int(fitted_model.predict(vector)[0])
    decision_score = float(fitted_model.decision_function(vector)[0])
    return {
        "status": "ANOMALY" if prediction == -1 else "NORMAL",
        "anomaly_score": -decision_score,
        "decision_score": decision_score,
        "is_anomaly": prediction == -1,
    }
