# KAIROS-AI

KAIROS-AI is a Streamlit prototype for acoustic machine-health monitoring. It turns a laptop microphone recording or an uploaded audio file into waveform, FFT, spectrogram, MFCC, and spectral features, then optionally applies a healthy-data IsolationForest model.

## Run on Windows 10/11

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Upload WAV/FLAC (and MP3 where the local SoundFile installation supports it), select a machine, and choose **Analyze audio**. Microphone recording needs a Windows default input device and microphone permission. No ESP32 hardware is required.

The main dashboard starts without importing NumPy or Pandas. Waveform and FFT use standard-library fallbacks; spectrograms, MFCC/spectral features, and IsolationForest remain feature-local and show an informative warning if Windows Application Control blocks a scientific dependency.

## Train the demonstration model

Place at least two healthy WAV/FLAC/OGG recordings in `dataset/healthy/`, then use **Train from dataset** in AI Diagnostics. The saved model is `models/isolation_forest.joblib`. The model only detects similarity to the healthy examples; it does not diagnose a bearing, belt, or motor fault and its score is not a failure probability.

## Architecture

- `app.py`: dashboard, navigation, session history, and charts.
- `modules/audio_input.py`: upload decoding, validation, preprocessing, and microphone capture.
- `modules/signal_processing.py`: waveform, FFT, spectrogram, and acoustic features.
- `modules/anomaly_detection.py`: persistence-backed IsolationForest with honest no-model fallback.
- `modules/machine_manager.py`: machine registry and session-derived snapshots.

## Limitations and future integration

The initial model is a demonstration anomaly detector and needs representative healthy recordings. There is no fake live sensor stream and no fault classification. A future ESP32 DevKit + INMP441 integration can publish compressed audio windows and metadata over MQTT or HTTP; a separate ingestion adapter can then feed the same preprocessing and analysis functions.

