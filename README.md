# KAIROS-AI

KAIROS-AI is a Streamlit prototype for acoustic machine-health monitoring. It
supports laptop microphone recordings and uploaded audio, and displays waveform,
FFT, spectrogram, MFCC, spectral features, and optional Isolation Forest results.
No live IoT feed, fault diagnosis, or performance claims are included.

The current prototype has been trained locally on 21 healthy fan `id_00`
recordings (the 20 repository samples plus one additional normal clip). Twenty
abnormal `id_00` recordings are kept separately for evaluation. With this small
sample set, the initial evaluation flagged 12 of 20 abnormal clips; this is a
development check, not a production accuracy claim.

## Installation and dashboard

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Upload WAV/FLAC (and MP3 where the local SoundFile installation supports it),
select a machine, and choose **Analyze audio**. The Acoustic Analysis page
shows the waveform, FFT spectrum, spectrogram, extracted features, and model
diagnostic result. AI Diagnostics shows model status and can train from the
healthy dataset. Machine Monitoring and Alerts & History provide the prototype
fleet view and session history. Microphone recording needs a default input
device and microphone permission.

## Anomaly detection module

The reusable Member 3 pipeline is in `modules/`. It converts WAV input to mono
at 22,050 Hz and extracts 13 MFCC means and standard deviations plus mean and
standard deviation summaries for RMS energy, spectral centroid, bandwidth,
rolloff, and zero-crossing rate. Training uses only healthy recordings and
saves a StandardScaler + Isolation Forest pipeline with Joblib.

Add real WAV files here. Train with `normal/` only; reserve `abnormal/` for
evaluation:

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

## Live monitoring and email alerts

After training `models/isolation_forest.joblib`, continuous monitoring can run
as a separate process from the Streamlit dashboard. It captures 5-second audio
windows from the default microphone, analyzes every 5 seconds, stores events in
`data/kairos_events.db`, and sends an email after three consecutive anomaly
windows. Email alerts are suppressed for one hour after an alert.

Configure SMTP through environment variables before starting the monitor:

```bash
export SMTP_HOST=smtp.example.com
export SMTP_PORT=587
export SMTP_USERNAME=your-user
export SMTP_PASSWORD=your-password
export ALERT_FROM=alerts@example.com
export ALERT_TO=you@example.com
python -m modules.live_monitor --machine-id MTR-01
```

Use `--no-email` to test microphone capture and event logging without SMTP.
The monitor is intended to run as a background service; closing the Streamlit
browser does not stop it.

### Recommended demo

1. Train the model from the healthy WAV recordings in `dataset/normal/`.
2. Start the dashboard with `streamlit run app.py`.
3. Open **Live Monitoring**, select the target machine, and turn email alerts
   off for the first run.
4. Start monitoring and wait for the waveform to appear.
5. Speak or make a controlled sound near the microphone to see new event rows
   and charts. This demonstrates the pipeline, but is not a machine-fault test.
6. For an alert demonstration, temporarily use a test model/dataset or set
   **Anomalous windows before alert** to `1`; restore it to `3` afterward.
7. Enable email only after SMTP settings have been tested. Use **Clear live
   history** to reset the event table and waveform between demonstrations.

### External microphone

The current monitor uses the system default input device. An external USB
microphone can be connected and selected as the operating system's default
recording device before starting monitoring. For a multi-device setup, list
available PortAudio devices with:

```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```

Then pass the device index or exact device name to the worker:

```bash
python -m modules.live_monitor --device 2 --machine-id MTR-01
```

The same device selector can be exposed in the dashboard later. For reliable
24x7 use, prefer a USB microphone with a fixed device name, disable automatic
power saving, and keep the microphone physically consistent with the recordings
used to train the model.

## Audio and feature pipeline

`modules/audio_input.py` exposes `load_audio(path)` and
`record_audio(duration, sample_rate, channels)`. Both return an audio array and
sample rate that can be passed to `modules.signal_processing.process_audio()`.
The signal-processing layer converts multichannel audio to normalized mono,
then provides waveform and spectrogram Plotly figures, one-sided FFT data,
dominant frequency, 13 MFCC coefficients, RMS energy, zero-crossing rate,
spectral centroid, bandwidth, rolloff, and spectral flux. The feature extractor
keeps a fixed feature order so the training and prediction paths use the same
schema.

## Architecture

- `app.py`: dashboard, navigation, session history, and charts.
- `modules/audio_input.py`: upload decoding, preprocessing, and microphone capture.
- `modules/signal_processing.py`: waveform, FFT, spectrogram, and acoustic views.
- `modules/feature_extraction.py`: shared fixed-order ML feature extraction.
- `modules/anomaly_detection.py`: pipeline persistence and prediction adapters.
- `modules/train.py`, `modules/predict.py`, `modules/evaluation.py`: ML workflows.
- `modules/live_monitor.py`: continuous microphone monitoring worker.
- `modules/alerts.py`: SMTP notifications, alert cooldown, and event storage.
- `modules/machine_manager.py`: machine registry and session-derived snapshots.

## Tests

```bash
python -m unittest discover -s tests
```

The tests use synthetic audio and validate feature extraction, model behavior,
and invalid-input handling. A trained `models/isolation_forest.joblib` file is
generated locally and is intentionally not required in source control; train
it from the dataset after cloning the project.
