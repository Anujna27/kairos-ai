"""Audio preprocessing, visualization, and feature extraction for Kairos AI."""

from __future__ import annotations

from typing import Any

import librosa
import numpy as np
import plotly.graph_objects as go
import soundfile as sf


def _validate_audio(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
    """Validate audio and return a finite float32 array and positive sample rate."""
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    if audio is None:
        raise ValueError("audio cannot be None")
    signal = np.asarray(audio)
    if signal.size == 0:
        raise ValueError("audio cannot be empty")
    if signal.ndim not in (1, 2):
        raise ValueError("audio must be a mono array or a 2D multichannel array")
    if not np.issubdtype(signal.dtype, np.number):
        raise TypeError("audio must contain numeric samples")
    signal = signal.astype(np.float32, copy=False)
    if not np.all(np.isfinite(signal)):
        raise ValueError("audio contains NaN or infinite samples")
    # SoundFile decodes multichannel data as (samples, channels); accept either
    # orientation by treating the smaller dimension as the channel dimension.
    if signal.ndim == 2 and min(signal.shape) <= 8:
        if signal.shape[0] <= 8 and signal.shape[1] > 8:
            signal = signal.T
    if signal.ndim == 2 and signal.shape[1] > 8:
        raise ValueError("audio has too many channels (maximum supported: 8)")
    return signal, int(sample_rate)


def preprocess_audio(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
    """Convert audio to normalized mono float32 samples; preserve its sample rate."""
    signal, rate = _validate_audio(audio, sample_rate)
    if signal.ndim == 2:
        signal = np.mean(signal, axis=1, dtype=np.float32)
    peak = float(np.max(np.abs(signal)))
    if peak > 0:
        signal = signal / peak
    return signal.astype(np.float32, copy=False), rate


def generate_waveform(audio: np.ndarray, sample_rate: int) -> go.Figure:
    """Create an interactive Plotly amplitude-versus-time waveform."""
    signal, rate = preprocess_audio(audio, sample_rate)
    times = np.arange(signal.size, dtype=np.float64) / rate
    fig = go.Figure(go.Scatter(x=times, y=signal, mode="lines", name="Amplitude"))
    fig.update_layout(title="Audio Waveform", xaxis_title="Time (seconds)", yaxis_title="Amplitude", template="plotly_white")
    return fig


def perform_fft(audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
    """Return the one-sided frequency and magnitude spectrum of an audio signal."""
    signal, rate = preprocess_audio(audio, sample_rate)
    if signal.size < 2:
        raise ValueError("at least two samples are required for FFT analysis")
    # Windowing reduces spectral leakage. Scale by the window sum for amplitude.
    window = np.hanning(signal.size)
    spectrum = np.fft.rfft(signal * window)
    frequencies = np.fft.rfftfreq(signal.size, d=1.0 / rate)
    magnitudes = (2.0 * np.abs(spectrum) / max(float(window.sum()), 1e-12)).astype(np.float64)
    magnitudes[0] *= 0.5
    if signal.size % 2 == 0:
        magnitudes[-1] *= 0.5
    return {"frequencies": frequencies, "magnitudes": magnitudes}


def get_dominant_frequency(audio: np.ndarray, sample_rate: int) -> float:
    """Return the strongest non-DC FFT frequency in hertz (0 for silence)."""
    spectrum = perform_fft(audio, sample_rate)
    if spectrum["magnitudes"].size <= 1 or not np.any(spectrum["magnitudes"][1:]):
        return 0.0
    index = int(np.argmax(spectrum["magnitudes"][1:]) + 1)
    return float(spectrum["frequencies"][index])


def generate_spectrogram(audio: np.ndarray, sample_rate: int) -> go.Figure:
    """Create an interactive dB-scaled spectrogram using Librosa."""
    signal, rate = preprocess_audio(audio, sample_rate)
    if signal.size < 2:
        raise ValueError("at least two samples are required for a spectrogram")
    n_fft = min(2048, max(32, 2 ** int(np.floor(np.log2(signal.size)))))
    hop = max(1, n_fft // 4)
    stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop)
    db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    times = librosa.frames_to_time(np.arange(db.shape[1]), sr=rate, hop_length=hop)
    freqs = librosa.fft_frequencies(sr=rate, n_fft=n_fft)
    fig = go.Figure(go.Heatmap(x=times, y=freqs, z=db, colorscale="Viridis", colorbar={"title": "dB"}, zmin=-80, zmax=0))
    fig.update_layout(title="Audio Spectrogram", xaxis_title="Time (seconds)", yaxis_title="Frequency (Hz)", template="plotly_white")
    return fig


def extract_mfcc(audio: np.ndarray, sample_rate: int, n_mfcc: int = 13) -> dict[str, Any]:
    """Extract MFCCs and per-coefficient mean and standard deviation."""
    signal, rate = preprocess_audio(audio, sample_rate)
    if isinstance(n_mfcc, bool) or not isinstance(n_mfcc, int) or n_mfcc < 1:
        raise ValueError("n_mfcc must be a positive integer")
    n_fft = min(2048, max(32, 2 ** int(np.floor(np.log2(signal.size)))))
    coeffs = librosa.feature.mfcc(y=signal, sr=rate, n_mfcc=n_mfcc, n_fft=n_fft)
    means = np.mean(coeffs, axis=1)
    stds = np.std(coeffs, axis=1)
    return {"coefficients": coeffs, "mean": means, "std": stds}


def extract_acoustic_features(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    """Extract scalar time- and frequency-domain acoustic features."""
    signal, rate = preprocess_audio(audio, sample_rate)
    n_fft = min(2048, max(32, 2 ** int(np.floor(np.log2(signal.size)))))
    hop = max(1, n_fft // 4)
    rms = librosa.feature.rms(y=signal, frame_length=n_fft, hop_length=hop, center=True)[0]
    zcr = librosa.feature.zero_crossing_rate(signal, frame_length=n_fft, hop_length=hop)[0]
    centroid = librosa.feature.spectral_centroid(y=signal, sr=rate, n_fft=n_fft, hop_length=hop)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=rate, n_fft=n_fft, hop_length=hop)[0]
    rolloff = librosa.feature.spectral_rolloff(y=signal, sr=rate, n_fft=n_fft, hop_length=hop)[0]
    flux_spec = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop))
    normalized = flux_spec / np.maximum(np.sum(flux_spec, axis=0, keepdims=True), 1e-12)
    flux = np.sqrt(np.sum(np.diff(normalized, axis=1) ** 2, axis=0)) if normalized.shape[1] > 1 else np.array([0.0])
    stats = {"rms_energy": rms, "zero_crossing_rate": zcr, "spectral_centroid_hz": centroid,
             "spectral_bandwidth_hz": bandwidth, "spectral_rolloff_hz": rolloff, "spectral_flux": flux}
    return {name: float(np.mean(values)) for name, values in stats.items()}


def process_audio(audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
    """Preprocess audio, extract ML-ready numeric features, and return figures separately."""
    signal, rate = preprocess_audio(audio, sample_rate)
    mfcc = extract_mfcc(signal, rate)
    acoustic = extract_acoustic_features(signal, rate)
    feature_vector = {**acoustic}
    for index, value in enumerate(mfcc["mean"]):
        feature_vector[f"mfcc_{index + 1}_mean"] = float(value)
    for index, value in enumerate(mfcc["std"]):
        feature_vector[f"mfcc_{index + 1}_std"] = float(value)
    fft = perform_fft(signal, rate)
    return {
        "sample_rate": rate,
        "audio": signal,
        "features": feature_vector,
        "feature_vector": np.asarray(list(feature_vector.values()), dtype=np.float64),
        "dominant_frequency_hz": get_dominant_frequency(signal, rate),
        "fft": fft,
        "mfcc": mfcc,
        "figures": {"waveform": generate_waveform(signal, rate), "spectrogram": generate_spectrogram(signal, rate)},
    }

