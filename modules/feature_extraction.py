"""Audio feature extraction shared by training and inference."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Final

logger = logging.getLogger(__name__)

SAMPLE_RATE: Final[int] = 22_050
N_MFCC: Final[int] = 13
N_FFT: Final[int] = 2_048
HOP_LENGTH: Final[int] = 512
MIN_AUDIO_SAMPLES: Final[int] = N_FFT

FEATURE_NAMES: Final[tuple[str, ...]] = tuple(
    name
    for i in range(1, N_MFCC + 1)
    for name in (f"mfcc_{i:02d}_mean", f"mfcc_{i:02d}_std")
) + tuple(
    f"{feature}_{stat}"
    for feature in (
        "rms_energy",
        "spectral_centroid",
        "spectral_bandwidth",
        "spectral_rolloff",
        "zero_crossing_rate",
    )
    for stat in ("mean", "std")
)


class FeatureExtractionError(ValueError):
    """Raised when an audio file cannot produce valid features."""


def _scientific_modules() -> tuple[Any, Any]:
    try:
        import librosa
        import numpy as np
    except ImportError as exc:
        from .signal_processing import ScientificDependencyError

        raise ScientificDependencyError(
            "Audio feature extraction needs NumPy and Librosa."
        ) from exc
    return librosa, np


def _read_audio(path: Path, librosa: Any) -> tuple[Any, int]:
    return librosa.load(path, sr=SAMPLE_RATE, mono=True)


def _summarize_features(
    audio: Any, path_label: str, librosa: Any, np: Any
) -> dict[str, float]:
    if audio.size == 0:
        raise FeatureExtractionError(f"Audio file is empty: {path_label}")
    if audio.size < MIN_AUDIO_SAMPLES:
        raise FeatureExtractionError(
            f"Audio is too short ({audio.size} samples); at least "
            f"{MIN_AUDIO_SAMPLES} samples are required at {SAMPLE_RATE} Hz"
        )
    if not np.isfinite(audio).all():
        raise FeatureExtractionError(
            f"Audio contains non-finite samples: {path_label}"
        )

    try:
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=SAMPLE_RATE,
            n_mfcc=N_MFCC,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
        )
        frame_features = {
            "rms_energy": librosa.feature.rms(
                y=audio, frame_length=N_FFT, hop_length=HOP_LENGTH
            )[0],
            "spectral_centroid": librosa.feature.spectral_centroid(
                y=audio, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH
            )[0],
            "spectral_bandwidth": librosa.feature.spectral_bandwidth(
                y=audio, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH
            )[0],
            "spectral_rolloff": librosa.feature.spectral_rolloff(
                y=audio, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH
            )[0],
            "zero_crossing_rate": librosa.feature.zero_crossing_rate(
                y=audio, frame_length=N_FFT, hop_length=HOP_LENGTH
            )[0],
        }
    except Exception as exc:
        logger.exception("Feature extraction failed for %s", path_label)
        raise FeatureExtractionError(
            f"Could not extract features from audio file: {path_label}"
        ) from exc

    features: dict[str, float] = {}
    for index, coefficient in enumerate(mfcc, start=1):
        features[f"mfcc_{index:02d}_mean"] = float(np.mean(coefficient))
        features[f"mfcc_{index:02d}_std"] = float(np.std(coefficient))
    for name, values in frame_features.items():
        features[f"{name}_mean"] = float(np.mean(values))
        features[f"{name}_std"] = float(np.std(values))

    if tuple(features) != FEATURE_NAMES or not np.isfinite(
        list(features.values())
    ).all():
        raise FeatureExtractionError(
            f"Feature extraction produced invalid values for {path_label}"
        )
    return features


def extract_features_from_audio(
    audio: Any, sample_rate: int
) -> dict[str, float]:
    """Extract the same features from in-memory mono or multichannel samples."""
    if sample_rate <= 0:
        raise FeatureExtractionError("Audio sample rate must be positive.")
    librosa, np = _scientific_modules()
    try:
        samples = np.asarray(audio, dtype=float)
        if samples.ndim == 2:
            samples = np.mean(samples, axis=1)
        if samples.ndim != 1:
            raise FeatureExtractionError("Audio must be a mono or multichannel array.")
        if sample_rate != SAMPLE_RATE:
            samples = librosa.resample(
                samples, orig_sr=sample_rate, target_sr=SAMPLE_RATE
            )
    except FeatureExtractionError:
        raise
    except Exception as exc:
        raise FeatureExtractionError("Could not prepare in-memory audio.") from exc
    return _summarize_features(samples, "<in-memory audio>", librosa, np)


def extract_features(audio_path: str | Path) -> dict[str, float]:
    """Extract consistently ordered features from a WAV recording.

    Recordings are converted to mono and resampled to 22,050 Hz. All
    frame-based features are summarized by their mean and standard deviation.
    """
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {path}")

    librosa, np = _scientific_modules()
    try:
        audio, _ = _read_audio(path, librosa)
    except Exception as exc:
        logger.exception("Could not decode audio file %s", path)
        raise FeatureExtractionError(f"Could not read audio file: {path}") from exc
    return _summarize_features(audio, str(path), librosa, np)
