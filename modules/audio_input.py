"""Audio acquisition helpers for Kairos AI.

The functions in this module return the ``(audio, sample_rate)`` pair
consumed by ``modules.signal_processing``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def load_audio(file_path: str | Path, *, dtype: str = "float32") -> tuple[np.ndarray, int]:
    """Load an audio file and return samples plus its original sample rate.

    Parameters
    ----------
    file_path:
        Path to a sound file supported by SoundFile, such as WAV or FLAC.
    dtype:
        NumPy dtype used for returned samples.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    try:
        audio, sample_rate = sf.read(path, dtype=dtype, always_2d=False)
    except RuntimeError as error:
        raise ValueError(f"Could not decode audio file '{path}': {error}") from error
    audio = np.asarray(audio)
    if audio.size == 0:
        raise ValueError(f"Audio file is empty: {path}")
    return audio, int(sample_rate)


def record_audio(duration: float = 5.0, sample_rate: int = 16_000, channels: int = 1) -> tuple[np.ndarray, int]:
    """Record microphone audio and return samples plus the sample rate.

    Requires the optional ``sounddevice`` package and an available input
    device. The returned array is shaped ``(samples,)`` for mono or
    ``(samples, channels)`` for multichannel recording.
    """
    if not isinstance(duration, (int, float)) or isinstance(duration, bool) or duration <= 0:
        raise ValueError("duration must be a positive number of seconds")
    if not isinstance(sample_rate, (int, np.integer)) or isinstance(sample_rate, bool) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    if not isinstance(channels, (int, np.integer)) or isinstance(channels, bool) or channels <= 0:
        raise ValueError("channels must be a positive integer")
    try:
        import sounddevice as sd
    except ImportError as error:
        raise ImportError("Microphone recording requires sounddevice. Install it with: pip install sounddevice") from error

    frame_count = int(round(float(duration) * int(sample_rate)))
    try:
        recording = sd.rec(frame_count, samplerate=int(sample_rate), channels=int(channels), dtype="float32")
        sd.wait()
    except Exception as error:
        raise RuntimeError(f"Microphone recording failed: {error}") from error
    if channels == 1:
        recording = recording[:, 0]
    return np.asarray(recording, dtype=np.float32), int(sample_rate)
