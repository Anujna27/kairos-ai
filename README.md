# KAIROS-AI

KAIROS-AI is a Streamlit prototype for acoustic machine-health monitoring. It
supports laptop microphone recordings and uploaded audio, and displays waveform,
FFT, spectrogram, MFCC, spectral features, and optional Isolation Forest results.
No live IoT feed, fault diagnosis, or performance claims are included.

## Installation and dashboard

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Upload WAV/FLAC (and MP3 where the local SoundFile installation supports it),
select a machine, and choose **Analyze audio**. Microphone recording needs a
default input device and microphone permission.

## Anomaly detection module

The reusable Member 3 pipeline is in `modules/`. It converts WAV input to mono
at 22,050 Hz and extracts 13 MFCC means and standard deviations plus mean and
standard deviation summaries for RMS energy, spectral centroid, bandwidth,
rolloff, and zero-crossing rate. Training uses only healthy recordings and
saves a StandardScaler + Isolation Forest pipeline with Joblib.

Add real WAV files here:

```text
dataset/
├── normal/       # preferred training data and normal evaluation examples
├── abnormal/     # labeled examples reserved for evaluation
└── healthy/      # compatibility fallback for the existing dashboard
```

Put at least two valid recordings in `dataset/normal/`. If that directory has
no WAV files, `dataset/healthy/` is accepted for dashboard compatibility.
Abnormal recordings are never used for fitting.

### Train

```bash
python -m modules.train --dataset dataset --model models/isolation_forest.joblib
```

The default contamination is `auto`; it can be set with `--contamination 0.05`.
Only recordings from `normal/` (or the documented `healthy/` fallback) are used.

### Predict

```bash
python -m modules.predict path/to/recording.wav \
  --model models/isolation_forest.joblib
```

Python interfaces:

```python
from modules.anomaly_detection import predict_from_features
from modules.feature_extraction import extract_features
from modules.predict import predict_anomaly

features = extract_features("recording.wav")
result = predict_from_features(features)  # loads the default trained model
# Or: result = predict_anomaly("recording.wav")
```

Results contain `status` (`NORMAL` or `ANOMALY`), `is_anomaly`,
`decision_score`, and `anomaly_score` (the negative decision score). Scores
are uncalibrated and are not probabilities.

### Evaluate

```bash
python -m modules.evaluation \
  --dataset dataset \
  --model models/isolation_forest.joblib \
  --output evaluation_results.json
```

Evaluation reports accuracy, precision, recall, F1, confusion matrix, false
positive/negative rates, and ROC-AUC when both labeled classes are available.
It writes JSON metrics and a confusion-matrix PNG alongside the output file.

## Architecture

- `app.py`: dashboard, navigation, session history, and charts.
- `modules/audio_input.py`: upload decoding, preprocessing, and microphone capture.
- `modules/signal_processing.py`: waveform, FFT, spectrogram, and acoustic views.
- `modules/feature_extraction.py`: shared fixed-order ML feature extraction.
- `modules/anomaly_detection.py`: pipeline persistence and prediction adapters.
- `modules/train.py`, `modules/predict.py`, `modules/evaluation.py`: ML workflows.
- `modules/machine_manager.py`: machine registry and session-derived snapshots.

## Tests

```bash
python -m unittest discover -s tests
```
