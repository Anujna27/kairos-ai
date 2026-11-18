"""Audio loading, validation, preprocessing, and optional microphone capture."""

from __future__ import annotations

import io
from typing import Any, BinaryIO, Tuple

SUPPORTED_FORMATS = {"wav", "flac", "mp3", "ogg"}


def _soundfile():
    try:
        import soundfile as sf
        return sf
    except ImportError as exc:
        raise RuntimeError("Audio decoding is unavailable because SoundFile could not be imported.") from exc


def load_audio(source: str | bytes | BinaryIO) -> Tuple[list[float], int]:
    """Load an audio source and return mono samples as a Python list plus rate."""
    sf = _soundfile()
    try:
        if isinstance(source, (bytes, bytearray)):
            data, sample_rate = sf.read(io.BytesIO(source), always_2d=True)
        else:
            data, sample_rate = sf.read(source, always_2d=True)
    except Exception as exc:
        raise ValueError("Could not decode this audio file. WAV and FLAC work reliably; MP3 support depends on local SoundFile/libsndfile support.") from exc
    rows = data.tolist() if hasattr(data, "tolist") else list(data)
    mono = [sum(float(value) for value in row) / len(row) if isinstance(row, list) else float(row) for row in rows]
    return preprocess_audio(mono, int(sample_rate))


def preprocess_audio(audio: Any, sample_rate: int) -> Tuple[list[float], int]:
    """Validate, mono-convert, clean, and peak-normalize using built-ins."""
    if sample_rate <= 0:
        raise ValueError("The audio sample rate is invalid.")
    values = audio.tolist() if hasattr(audio, "tolist") else list(audio)
    if values and isinstance(values[0], list):
        values = [sum(float(value) for value in row) / len(row) for row in values]
    samples = [0.0 if value != value else float(value) for value in values]
    if len(samples) < 32 or not any(abs(value) > 1e-8 for value in samples):
        raise ValueError("The recording is empty or too quiet to analyze.")
    peak = max(abs(value) for value in samples)
    if peak > 1.0:
        samples = [value / peak for value in samples]
    return samples, int(sample_rate)


def record_audio(duration: float = 5.0, sample_rate: int = 16_000) -> Tuple[list[float], int]:
    """Record from the default laptop microphone using sounddevice."""
    if duration <= 0 or sample_rate <= 0:
        raise ValueError("Duration and sample rate must be positive.")
    try:
        import sounddevice as sd
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
        sd.wait()
    except Exception as exc:
        raise RuntimeError("Microphone recording failed. Check that a default input device is connected and Windows has microphone permission enabled.") from exc
    return preprocess_audio(recording[:, 0], sample_rate)

