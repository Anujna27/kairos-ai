# Kairos AI signal processing

This module provides reusable preprocessing, Plotly visualization, FFT, MFCC, and acoustic feature extraction. It does not implement anomaly detection.

## Use

```python
from modules.signal_processing import process_audio

result = process_audio(audio_array, sample_rate)
model_features = result["features"]       # named numeric values for sklearn
model_vector = result["feature_vector"]   # NumPy vector in matching insertion order
waveform = result["figures"]["waveform"]
spectrogram = result["figures"]["spectrogram"]
```

`audio_array` may be mono `(samples,)` or multichannel `(samples, channels)` audio. The returned audio is normalized mono float32; sample rate is preserved. `generate_waveform`, `perform_fft`, `get_dominant_frequency`, `generate_spectrogram`, `extract_mfcc`, and `extract_acoustic_features` can also be called independently. MFCC extraction defaults to 13 coefficients. Invalid, empty, non-finite, or unsupported input raises `ValueError` or `TypeError` with a clear message.

## Run the synthetic checks

From the project directory, run `python -m tests.test_signal_processing`. The checks use a 440 Hz tone and validate the processing functions and input handling.
