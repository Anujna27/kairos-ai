"""Focused tests for feature handling and anomaly model behavior."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import numpy as np
    import pandas as pd

    from modules.anomaly_detection import (
        create_pipeline,
        load_model,
        predict_from_features,
        save_model,
    )
    from modules.feature_extraction import (
        FEATURE_NAMES,
        MIN_AUDIO_SAMPLES,
        FeatureExtractionError,
        extract_features,
        extract_features_from_audio,
    )
    from modules.evaluation import evaluate_model
    from modules.train import train_model
    SCIENTIFIC_DEPENDENCIES_AVAILABLE = True
    SCIENTIFIC_DEPENDENCY_ERROR = ""
except (ImportError, OSError) as exc:
    SCIENTIFIC_DEPENDENCIES_AVAILABLE = False
    SCIENTIFIC_DEPENDENCY_ERROR = str(exc)


@unittest.skipUnless(SCIENTIFIC_DEPENDENCIES_AVAILABLE, SCIENTIFIC_DEPENDENCY_ERROR)
class FeatureExtractionTests(unittest.TestCase):
    def test_features_have_stable_named_order(self) -> None:
        audio = np.zeros(MIN_AUDIO_SAMPLES, dtype=float)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mock.wav"
            path.touch()
            with patch("modules.feature_extraction._read_audio", return_value=(audio, 22_050)):
                features = extract_features(path)

        self.assertEqual(tuple(features), FEATURE_NAMES)
        self.assertEqual(len(features), 36)
        self.assertTrue(all(np.isfinite(value) for value in features.values()))

    def test_rejects_too_short_recording(self) -> None:
        audio = np.zeros(MIN_AUDIO_SAMPLES - 1, dtype=float)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mock.wav"
            path.touch()
            with patch("modules.feature_extraction._read_audio", return_value=(audio, 22_050)):
                with self.assertRaises(FeatureExtractionError):
                    extract_features(path)

    def test_in_memory_features_use_same_schema(self) -> None:
        audio = np.zeros(MIN_AUDIO_SAMPLES, dtype=float)
        features = extract_features_from_audio(audio, 22_050)
        self.assertEqual(tuple(features), FEATURE_NAMES)


@unittest.skipUnless(SCIENTIFIC_DEPENDENCIES_AVAILABLE, SCIENTIFIC_DEPENDENCY_ERROR)
class AnomalyDetectionTests(unittest.TestCase):
    @staticmethod
    def _features(offset: float = 0.0) -> dict[str, float]:
        return {name: float(index + offset) for index, name in enumerate(FEATURE_NAMES)}

    def test_predict_returns_convention_and_unscaled_scores(self) -> None:
        training = pd.DataFrame(
            [
                [index + offset for index in range(len(FEATURE_NAMES))]
                for offset in (0, 1, 2, 3)
            ],
            columns=FEATURE_NAMES,
        )
        model = create_pipeline(random_state=7).fit(training)

        prediction = predict_from_features(self._features(0.5), model)

        self.assertIn(prediction["status"], {"NORMAL", "ANOMALY"})
        self.assertEqual(prediction["is_anomaly"], prediction["status"] == "ANOMALY")
        self.assertAlmostEqual(
            float(prediction["anomaly_score"]), -float(prediction["decision_score"])
        )
        self.assertEqual(
            prediction,
            predict_from_features(dict(reversed(list(self._features(0.5).items()))), model),
        )

    def test_model_round_trip(self) -> None:
        training = pd.DataFrame(
            [
                [index + offset for index in range(len(FEATURE_NAMES))]
                for offset in (0, 1, 2)
            ],
            columns=FEATURE_NAMES,
        )
        model = create_pipeline().fit(training)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            save_model(model, path)
            loaded = load_model(path)

        self.assertEqual(
            predict_from_features(self._features(0.25), model),
            predict_from_features(self._features(0.25), loaded),
        )


@unittest.skipUnless(SCIENTIFIC_DEPENDENCIES_AVAILABLE, SCIENTIFIC_DEPENDENCY_ERROR)
class TrainingTests(unittest.TestCase):
    def test_training_uses_normal_and_supports_healthy_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            normal_dir = root / "normal"
            abnormal_dir = root / "abnormal"
            normal_dir.mkdir()
            abnormal_dir.mkdir()
            (normal_dir / "normal_a.wav").touch()
            (normal_dir / "normal_b.wav").touch()
            (abnormal_dir / "abnormal.wav").touch()

            def fake_features(path: Path) -> dict[str, float]:
                offset = 1.0 if path.stem.endswith("b") else 0.0
                return {
                    name: float(index + offset)
                    for index, name in enumerate(FEATURE_NAMES)
                }

            normal_model_path = root / "normal-model.joblib"
            with patch(
                "modules.train.extract_features", side_effect=fake_features
            ) as extract:
                model = train_model(root, model_path=normal_model_path)
            self.assertTrue(normal_model_path.is_file())
            self.assertEqual(model.named_steps["scaler"].n_samples_seen_, 2)
            self.assertEqual(
                {call.args[0].parent.name for call in extract.call_args_list},
                {"normal"},
            )

            healthy_root = root / "legacy-dataset"
            healthy_dir = healthy_root / "healthy"
            healthy_dir.mkdir(parents=True)
            (healthy_dir / "healthy_a.wav").touch()
            (healthy_dir / "healthy_b.wav").touch()
            with patch("modules.train.extract_features", side_effect=fake_features):
                model = train_model(
                    healthy_root, model_path=root / "healthy-model.joblib"
                )
            self.assertEqual(model.named_steps["scaler"].n_samples_seen_, 2)


@unittest.skipUnless(SCIENTIFIC_DEPENDENCIES_AVAILABLE, SCIENTIFIC_DEPENDENCY_ERROR)
class EvaluationTests(unittest.TestCase):
    def test_metrics_and_saved_outputs(self) -> None:
        predictions = {
            "normal/normal.wav": (False, 0.1),
            "normal/false_positive.wav": (True, 0.8),
            "abnormal/abnormal.wav": (True, 0.9),
            "abnormal/missed.wav": (False, 0.2),
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            normal_dir = root / "normal"
            abnormal_dir = root / "abnormal"
            normal_dir.mkdir()
            abnormal_dir.mkdir()
            for path in (
                normal_dir / "normal.wav",
                normal_dir / "false_positive.wav",
                abnormal_dir / "abnormal.wav",
                abnormal_dir / "missed.wav",
            ):
                path.touch()

            def fake_predict(path: Path, _model: object) -> dict[str, object]:
                is_anomaly, anomaly_score = predictions[
                    f"{path.parent.name}/{path.name}"
                ]
                return {
                    "is_anomaly": is_anomaly,
                    "anomaly_score": anomaly_score,
                }

            output_path = root / "results.json"
            with patch("modules.evaluation.predict_anomaly", side_effect=fake_predict):
                result = evaluate_model(
                    create_pipeline(), normal_dir, abnormal_dir, output_path
                )

            self.assertEqual(result["confusion_matrix"], [[1, 1], [1, 1]])
            self.assertEqual(result["accuracy"], 0.5)
            self.assertEqual(result["false_positive_rate"], 0.5)
            self.assertEqual(result["false_negative_rate"], 0.5)
            self.assertEqual(result["roc_auc"], 0.75)
            saved = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["evaluated_recordings"], 4)
            self.assertTrue(Path(result["confusion_matrix_plot"]).is_file())


if __name__ == "__main__":
    unittest.main()
