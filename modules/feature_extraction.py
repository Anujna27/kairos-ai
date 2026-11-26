"""Audio feature extraction shared by training and inference."""

from __future__ import annotations

import logging
import math
import statistics
from pathlib import Path
from typing import Any, Final

from .signal_processing import ScientificDependencyError

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


def _resample_linear(samples: list[float], source_rate: int, target_rate: int) -> list[float]:
    """Small dependency-free linear resampler for the restricted fallback path."""
    if source_rate == target_rate or not samples:
        return samples
    output_length = max(1, round(len(samples) * target_rate / source_rate))
    scale = (len(samples) - 1) / max(output_length - 1, 1)
    output = []
    for index in range(output_length):
        position = index * scale
        left = min(int(position), len(samples) - 1)
        right = min(left + 1, len(samples) - 1)
        fraction = position - left
        output.append(samples[left] * (1 - fraction) + samples[right] * fraction)
    return output


def _fallback_features(audio: Any, sample_rate: int, path_label: str) -> dict[str, float]:
    """Extract transparent approximate features without NumPy, Librosa, or Numba."""
    values = audio.tolist() if hasattr(audio, "tolist") else list(audio)
    if values and isinstance(values[0], list):
        values = [sum(float(value) for value in row) / len(row) for row in values]
    samples = [float(value) for value in values]
    if len(samples) < MIN_AUDIO_SAMPLES:
        raise FeatureExtractionError(
            f"Audio is too short ({len(samples)} samples); at least {MIN_AUDIO_SAMPLES} samples are required at {SAMPLE_RATE} Hz"
        )
    if not all(math.isfinite(value) for value in samples):
        raise FeatureExtractionError(f"Audio contains non-finite samples: {path_label}")
    samples = _resample_linear(samples, sample_rate, SAMPLE_RATE)
    frames = []
    for start in range(0, max(1, len(samples) - N_FFT + 1), HOP_LENGTH):
        frame = samples[start : start + N_FFT]
        if len(frame) < N_FFT:
            frame = frame + [0.0] * (N_FFT - len(frame))
        frames.append(frame)
    if not frames:
        frames = [samples[:N_FFT] + [0.0] * max(0, N_FFT - len(samples))]

    mfcc_frames = [[] for _ in range(N_MFCC)]
    frame_features = {name: [] for name in ("rms_energy", "spectral_centroid", "spectral_bandwidth", "spectral_rolloff", "zero_crossing_rate")}
    for frame in frames:
        energy = [abs(value) for value in frame]
        total = sum(energy) or 1.0
        rms = math.sqrt(sum(value * value for value in frame) / len(frame))
        centroid = sum(index * value for index, value in enumerate(energy)) / total * (SAMPLE_RATE / (2 * len(frame)))
        bandwidth = math.sqrt(sum(((index * SAMPLE_RATE / (2 * len(frame))) - centroid) ** 2 * value for index, value in enumerate(energy)) / total)
        threshold = total * 0.85
        running = 0.0
        rolloff_index = 0
        for index, value in enumerate(energy):
            running += value
            if running >= threshold:
                rolloff_index = index
                break
        zcr = sum(1 for left, right in zip(frame, frame[1:]) if (left < 0 <= right) or (left >= 0 > right)) / max(len(frame) - 1, 1)
        frame_features["rms_energy"].append(rms)
        frame_features["spectral_centroid"].append(centroid)
        frame_features["spectral_bandwidth"].append(bandwidth)
        frame_features["spectral_rolloff"].append(rolloff_index * SAMPLE_RATE / (2 * len(frame)))
        frame_features["zero_crossing_rate"].append(zcr)
        segment_size = max(1, len(frame) // N_MFCC)
        log_energies = [math.log(sum(abs(value) for value in frame[index * segment_size : (index + 1) * segment_size]) / segment_size + 1e-8) for index in range(N_MFCC)]
        for coefficient in range(N_MFCC):
            mfcc_frames[coefficient].append(sum(log_energies[index] * math.cos(math.pi * coefficient * (index + 0.5) / N_MFCC) for index in range(N_MFCC)))

    features: dict[str, float] = {}
    for index, coefficient_values in enumerate(mfcc_frames, start=1):
        features[f"mfcc_{index:02d}_mean"] = float(statistics.fmean(coefficient_values))
        features[f"mfcc_{index:02d}_std"] = float(statistics.pstdev(coefficient_values))
    for name, values_for_feature in frame_features.items():
        features[f"{name}_mean"] = float(statistics.fmean(values_for_feature))
        features[f"{name}_std"] = float(statistics.pstdev(values_for_feature))
    return features


def _scientific_modules() -> tuple[Any, Any]:
    try:
        import librosa
        import numpy as np
    except (ImportError, OSError) as exc:
        raise ScientificDependencyError(
            "Audio feature extraction needs NumPy and Librosa. Windows Application Control "
            "is blocking a compiled NumPy component, so this feature is temporarily unavailable."
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
    except (ImportError, OSError) as exc:
        raise ScientificDependencyError(
            "Audio feature extraction is temporarily unavailable because Windows Application "
            "Control blocked a compiled NumPy/Numba component."
        ) from exc
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
    try:
        librosa, np = _scientific_modules()
    except ScientificDependencyError:
        return _fallback_features(audio, sample_rate, "<in-memory audio>")
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
    except (ImportError, OSError) as exc:
        raise ScientificDependencyError(
            "Audio preparation is temporarily unavailable because Windows Application "
            "Control blocked a compiled NumPy/Numba component."
        ) from exc
    except Exception as exc:
        raise FeatureExtractionError("Could not prepare in-memory audio.") from exc
    try:
        return _summarize_features(samples, "<in-memory audio>", librosa, np)
    except ScientificDependencyError:
        return _fallback_features(samples, sample_rate, "<in-memory audio>")


def extract_features(audio_path: str | Path) -> dict[str, float]:
    """Extract consistently ordered features from a WAV recording.

    Recordings are converted to mono and resampled to 22,050 Hz. All
    frame-based features are summarized by their mean and standard deviation.
    """
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {path}")

    try:
        librosa, np = _scientific_modules()
    except ScientificDependencyError:
        from .audio_input import load_audio

        audio, sample_rate = load_audio(str(path))
        return _fallback_features(audio, sample_rate, str(path))
    try:
        audio, _ = _read_audio(path, librosa)
    except Exception as exc:
        logger.exception("Could not decode audio file %s", path)
        raise FeatureExtractionError(f"Could not read audio file: {path}") from exc
    try:
        return _summarize_features(audio, str(path), librosa, np)
    except ScientificDependencyError:
        from .audio_input import load_audio

        fallback_audio, fallback_rate = load_audio(str(path))
        return _fallback_features(fallback_audio, fallback_rate, str(path))
