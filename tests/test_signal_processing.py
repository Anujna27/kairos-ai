"""Synthetic checks for Kairos AI signal processing."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from modules.signal_processing import (
    extract_acoustic_features, extract_mfcc, generate_spectrogram,
    generate_waveform, get_dominant_frequency, perform_fft, preprocess_audio,
)


def main():
    sample_rate = 16_000
    time = np.arange(sample_rate) / sample_rate
    audio = 0.6 * np.sin(2 * np.pi * 440 * time) + 0.2 * np.sin(2 * np.pi * 880 * time)
    assert abs(get_dominant_frequency(audio, sample_rate) - 440) < 2
    assert len(perform_fft(audio, sample_rate)["frequencies"]) > 0
    assert len(generate_waveform(audio, sample_rate).data) > 0
    assert len(generate_spectrogram(audio, sample_rate).data) > 0
    assert len(extract_mfcc(audio, sample_rate)["mean"]) == 13
    assert "rms_energy" in extract_acoustic_features(audio, sample_rate)
    stereo = np.column_stack((audio, audio))
    mono, rate = preprocess_audio(stereo, sample_rate)
    assert mono.ndim == 1 and rate == sample_rate and np.isclose(np.max(np.abs(mono)), 1)
    for invalid, invalid_rate in ((np.array([]), sample_rate), (audio, 0), (np.array([np.nan]), sample_rate)):
        try:
            preprocess_audio(invalid, invalid_rate)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid input should raise ValueError")
    print("All signal processing checks passed.")


if __name__ == "__main__":
    main()
