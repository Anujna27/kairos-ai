"""Audio acquisition helpers for EchoSense.

The public functions return mono, normalized ``float32`` samples and the
original sample rate, ready for the signal-processing and anomaly modules.
Audio is not resampled here so downstream frequency calculations remain
correct for the source recording.
"""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import numpy as np


AudioSource = str | Path | BinaryIO


def preprocess_audio(audio: np.ndarray, *, normalize: bool = True) -> np.ndarray:
    """Convert audio to mono float32 and optionally peak-normalize it.

    Multichannel audio is averaged across channels. Silence remains silence;
    non-finite samples are replaced with zero. Peak normalization preserves
    relative waveform shape but removes absolute loudness information.
    """
    samples = np.asarray(audio)
    if samples.size == 0:
        raise ValueError("Audio is empty.")

    # SoundFile returns (frames, channels); also support channel-first arrays.
    if samples.ndim == 2:
        if samples.shape[1] <= 8:
            samples = samples.mean(axis=1)
        elif samples.shape[0] <= 8:
            samples = samples.mean(axis=0)
        else:
            raise ValueError("Audio must be mono or have a small number of channels.")
    elif samples.ndim != 1:
        raise ValueError("Audio must be a one-dimensional signal or a 2D channel array.")

    samples = np.nan_to_num(samples.astype(np.float32, copy=False), copy=True)
    if normalize:
        peak = float(np.max(np.abs(samples)))
        if peak > 0:
            samples /= peak
    return samples


def load_audio(source: AudioSource, *, normalize: bool = True) -> tuple[np.ndarray, int]:
    """Read a WAV/audio file or Streamlit UploadedFile.

    Returns ``(audio, sample_rate)``. ``source`` may be a path or a binary
    file-like object (including ``st.file_uploader`` output). Requires
    ``soundfile``; unsupported/corrupt files raise a descriptive ValueError.
    """
    try:
        import soundfile as sf
    except ImportError as exc:
        raise ImportError("Reading uploaded audio requires soundfile (pip install soundfile).") from exc

    try:
        if hasattr(source, "seek"):
            source.seek(0)
        samples, sample_rate = sf.read(source, dtype="float32", always_2d=False)
    except Exception as exc:
        raise ValueError(f"Could not read audio file: {exc}") from exc

    if sample_rate <= 0:
        raise ValueError("Audio file has an invalid sample rate.")
    return preprocess_audio(samples, normalize=normalize), int(sample_rate)


def record_audio(
    duration: float = 5.0,
    sample_rate: int = 22_050,
    *,
    normalize: bool = True,
    device: int | str | None = None,
) -> tuple[np.ndarray, int]:
    """Record mono audio from the default (or selected) microphone.

    Requires ``sounddevice``. Microphone access is requested only when this
    function is called, so importing this module works without audio hardware.
    """
    if duration <= 0:
        raise ValueError("Recording duration must be greater than zero.")
    if sample_rate <= 0:
        raise ValueError("Sample rate must be greater than zero.")
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise ImportError("Microphone recording requires sounddevice (pip install sounddevice).") from exc

    try:
        frames = int(round(duration * sample_rate))
        recording = sd.rec(
            frames,
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            device=device,
        )
        sd.wait()
    except Exception as exc:
        raise RuntimeError(
            "Microphone recording failed. Check that a microphone is connected, "
            "available to the operating system, and not in use by another app."
        ) from exc

    return preprocess_audio(recording, normalize=normalize), sample_rate


def save_wav(
    path: str | Path,
    audio: np.ndarray,
    sample_rate: int,
) -> Path:
    """Save a mono or multichannel NumPy signal as a WAV file."""
    if sample_rate <= 0:
        raise ValueError("Sample rate must be greater than zero.")
    try:
        import soundfile as sf
    except ImportError as exc:
        raise ImportError("Saving WAV files requires soundfile (pip install soundfile).") from exc

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(target, np.asarray(audio), sample_rate, format="WAV")
    return target


__all__ = ["load_audio", "preprocess_audio", "record_audio", "save_wav"]
