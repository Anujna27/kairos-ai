"""Isolation Forest model construction, persistence, and inference."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .feature_extraction import FEATURE_NAMES
from .signal_processing import ScientificDependencyError

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "isolation_forest.joblib"
MODEL_PATH = DEFAULT_MODEL_PATH


def _model_components() -> tuple[Any, ...]:
    try:
        import joblib
        import numpy as np
        import pandas as pd
        from sklearn.ensemble import IsolationForest
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except (ImportError, OSError) as exc:
        raise ScientificDependencyError(
            "AI anomaly detection needs NumPy, Pandas, Joblib, and Scikit-learn."
        ) from exc
    return joblib, np, pd, IsolationForest, Pipeline, StandardScaler


def create_pipeline(
    contamination: float | str = "auto", random_state: int = 42
) -> Pipeline:
    """Create the scaled Isolation Forest pipeline."""
    _, _, _, IsolationForest, Pipeline, StandardScaler = _model_components()
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
    joblib, *_ = _model_components()
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Saved anomaly detection model to %s", path)
    return path


def load_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> Pipeline:
    """Load a persisted fitted pipeline, raising explicit errors on failure."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Model not found at {path}. Train a model before making predictions."
        )
    joblib, _, _, _, Pipeline, _ = _model_components()
    model: Any = joblib.load(path)
    if not isinstance(model, Pipeline) or "isolation_forest" not in model.named_steps:
        raise ValueError(f"File does not contain an EchoSense Isolation Forest: {path}")
    return model


def _ordered_feature_vector(features: dict[str, float]) -> Any:
    """Validate and order a feature mapping according to the training schema."""
    _, np, pd, *_ = _model_components()
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


def model_status(model_path: str | Path = MODEL_PATH) -> dict[str, str]:
    """Return a dashboard-friendly model/dependency status."""
    try:
        _model_components()
    except ScientificDependencyError as exc:
        return {"status": "UNAVAILABLE", "detail": str(exc)}
    path = Path(model_path)
    if not path.is_file():
        return {
            "status": "MODEL NOT TRAINED",
            "detail": "Add healthy WAV recordings under dataset/normal and train the model.",
        }
    return {"status": "TRAINED", "detail": "Isolation Forest model is available."}


def train_from_directory(dataset_dir: str | Path) -> dict[str, object]:
    """Train the dashboard model from healthy WAVs in the dataset directory."""
    from .train import train_model

    model = train_model(dataset_dir, model_path=MODEL_PATH)
    samples = int(model.named_steps["scaler"].n_samples_seen_)
    return {
        "path": str(MODEL_PATH),
        "samples": samples,
        "algorithm": "IsolationForest",
    }


def predict_anomaly(
    features: dict[str, float], model: Pipeline | None = None
) -> dict[str, Any]:
    """Dashboard adapter retaining its score/explanation fields."""
    if model is None:
        status = model_status()
        if status["status"] != "TRAINED":
            return {
                **status,
                "score": None,
                "explanation": status["detail"],
                "anomaly_score": None,
                "decision_score": None,
                "is_anomaly": False,
            }
        model = load_model()

    result = predict_from_features(features, model)
    return {
        **result,
        "score": result["decision_score"],
        "explanation": (
            "Relative Isolation Forest score; this is not a failure probability "
            "or a fault classification."
        ),
    }
