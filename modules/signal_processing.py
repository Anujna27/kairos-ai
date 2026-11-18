"""Acoustic calculations with standard-library fallbacks and lazy scientific imports."""

from __future__ import annotations

import cmath
import math
from typing import Any, Dict, Tuple


class ScientificDependencyError(RuntimeError):
    """Raised when an optional NumPy/Librosa feature is blocked or unavailable."""


def _scientific_modules():
    try:
        import librosa
        import numpy as np
        return librosa, np
    except ImportError as exc:
        raise ScientificDependencyError("This feature needs NumPy and Librosa. Windows Application Control is blocking a NumPy compiled component, so this feature is temporarily unavailable.") from exc


def generate_waveform(audio: Any, sample_rate: int) -> Tuple[list[float], list[float]]:
    """Return time and amplitude lists for a waveform chart."""
    samples = audio.tolist() if hasattr(audio, "tolist") else list(audio)
    return [index / sample_rate for index in range(len(samples))], [float(value) for value in samples]


def calculate_fft(audio: Any, sample_rate: int) -> Tuple[list[float], list[float], float]:
    """Calculate a pure-Python positive-frequency FFT for fallback use."""
    samples = audio.tolist() if hasattr(audio, "tolist") else [float(value) for value in audio]
    samples = samples[: min(len(samples), 8192)]
    n = len(samples)
    if not n:
        return [], [], 0.0
    windowed = [sample * (0.5 - 0.5 * math.cos(2 * math.pi * index / max(n - 1, 1))) for index, sample in enumerate(samples)]
    bins = n // 2 + 1
    magnitudes = [abs(sum(windowed[t] * cmath.exp(-2j * math.pi * k * t / n) for t in range(n))) / n for k in range(bins)]
    frequencies = [k * sample_rate / n for k in range(bins)]
    dominant = frequencies[magnitudes.index(max(magnitudes))] if magnitudes else 0.0
    return frequencies, magnitudes, float(dominant)


def generate_spectrogram(audio: Any, sample_rate: int) -> Tuple[Any, Any, Any]:
    """Generate a Librosa spectrogram; dependency errors remain feature-local."""
    librosa, np = _scientific_modules()
    samples = np.asarray(audio, dtype=np.float32)
    n_fft = min(2048, max(256, 2 ** int(np.floor(np.log2(max(len(samples), 256))))))
    hop = max(64, n_fft // 4)
    spectrogram = np.abs(librosa.stft(samples, n_fft=n_fft, hop_length=hop))
    db = librosa.amplitude_to_db(spectrogram + 1e-8, ref=np.max)
    return db, librosa.frames_to_time(np.arange(db.shape[1]), sr=sample_rate, hop_length=hop), librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)


def extract_features(audio: Any, sample_rate: int) -> Dict[str, float]:
    """Extract MFCC and spectral features; unavailable dependencies are explicit."""
    librosa, np = _scientific_modules()
    y = np.asarray(audio, dtype=np.float32)
    rms = float(np.mean(librosa.feature.rms(y=y)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sample_rate)))
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sample_rate)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    mfcc = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=13)
    features = {f"mfcc_{i + 1}": float(np.mean(mfcc[i])) for i in range(mfcc.shape[0])}
    features.update({"rms_energy": rms, "spectral_centroid_hz": centroid, "spectral_bandwidth_hz": bandwidth, "zero_crossing_rate": zcr})
    return features


def feature_vector(features: Dict[str, float]) -> Any:
    """Create a stable NumPy vector only when model work needs it."""
    try:
        import numpy as np
    except ImportError as exc:
        raise ScientificDependencyError("AI feature vectors need NumPy, but Windows Application Control is blocking a NumPy component.") from exc
    keys = [f"mfcc_{i}" for i in range(1, 14)] + ["rms_energy", "spectral_centroid_hz", "spectral_bandwidth_hz", "zero_crossing_rate"]
    return np.asarray([features.get(key, 0.0) for key in keys], dtype=np.float32)

