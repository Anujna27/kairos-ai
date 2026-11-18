"""Persistence-backed IsolationForest detection with explicit dependency status."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .signal_processing import ScientificDependencyError, extract_features, feature_vector


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "isolation_forest.joblib"


def _model_dependencies():
    try:
        import joblib
        import numpy as np
        from sklearn.ensemble import IsolationForest
        return joblib, np, IsolationForest
    except ImportError as exc:
        raise ScientificDependencyError("AI anomaly detection needs NumPy and Scikit-learn. Windows Application Control is blocking a NumPy compiled component, so AI predictions are temporarily unavailable.") from exc


def model_status(model_path: Path = MODEL_PATH) -> Dict[str, str]:
    """Return a UI-safe status without importing NumPy or Scikit-learn at startup."""
    try:
        _model_dependencies()
    except ScientificDependencyError as exc:
        return {"status": "UNAVAILABLE", "detail": str(exc)}
    if not model_path.exists():
        return {"status": "MODEL NOT TRAINED", "detail": "Add healthy recordings to dataset/healthy and train the model."}
    return {"status": "TRAINED", "detail": "IsolationForest model is available."}


def train_model(feature_vectors: Iterable[Any], model_path: Path = MODEL_PATH) -> Dict[str, object]:
    """Train on healthy feature vectors only; no accuracy claim is produced."""
    joblib, np, IsolationForest = _model_dependencies()
    matrix = np.asarray(list(feature_vectors), dtype=np.float32)
    if matrix.ndim != 2 or len(matrix) < 2:
        raise ValueError("At least two healthy recordings are required to train the demonstration model.")
    model = IsolationForest(n_estimators=150, contamination="auto", random_state=42)
    model.fit(matrix)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return {"path": str(model_path), "samples": int(len(matrix)), "algorithm": "IsolationForest"}


def train_from_directory(dataset_dir: Path) -> Dict[str, object]:
    """Extract one feature vector per healthy audio file and train the model."""
    from .audio_input import load_audio
    paths = [p for p in (dataset_dir / "healthy").glob("**/*") if p.suffix.lower().lstrip(".") in {"wav", "flac", "ogg"}]
    vectors = []
    for path in paths:
        try:
            audio, rate = load_audio(str(path))
            vectors.append(feature_vector(extract_features(audio, rate)))
        except (ValueError, ScientificDependencyError):
            continue
    return train_model(vectors)


def save_model(model: Any, model_path: Path = MODEL_PATH) -> None:
    joblib, _, _ = _model_dependencies()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


def load_model(model_path: Path = MODEL_PATH) -> Optional[Any]:
    """Load a model if dependencies and the persisted model are available."""
    if not model_path.exists():
        return None
    try:
        joblib, _, _ = _model_dependencies()
        return joblib.load(model_path)
    except (ImportError, OSError, ValueError, TypeError):
        return None


def predict_anomaly(features: Dict[str, float], model: Optional[Any] = None) -> Dict[str, object]:
    """Predict NORMAL/ANOMALY, or explicitly report unavailable model state."""
    if model is None:
        status = model_status()
        if status["status"] == "UNAVAILABLE":
            return {"status": "AI UNAVAILABLE", "score": None, "explanation": status["detail"]}
        return {"status": "MODEL NOT TRAINED", "score": None, "explanation": "Add healthy recordings to dataset/healthy and train the model."}
    vector = feature_vector(features).reshape(1, -1)
    decision = float(model.decision_function(vector)[0])
    prediction = int(model.predict(vector)[0])
    return {"status": "NORMAL" if prediction == 1 else "ANOMALY", "score": decision, "explanation": "Relative IsolationForest score; this is not a failure probability or fault classification."}

