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

Live monitoring can notify up to three email recipients. Enter the SMTP
settings and one to three recipient addresses on the Live Monitoring page;
the same alert is delivered to each address.

SMTP values can be saved locally as Windows user environment variables so they
do not need to be entered each time the dashboard starts:

```powershell
[Environment]::SetEnvironmentVariable("SMTP_HOST", "smtp.gmail.com", "User")
[Environment]::SetEnvironmentVariable("SMTP_PORT", "587", "User")
[Environment]::SetEnvironmentVariable("SMTP_USERNAME", "kairos.machine.alerts@gmail.com", "User")
[Environment]::SetEnvironmentVariable("SMTP_PASSWORD", "your-gmail-app-password", "User")
[Environment]::SetEnvironmentVariable("ALERT_FROM", "kairos.machine.alerts@gmail.com", "User")
[Environment]::SetEnvironmentVariable("ALERT_TO", "person1@example.com,person2@example.com,person3@example.com", "User")
```

Restart PowerShell and Streamlit after changing environment variables. Use a
Gmail App Password for `SMTP_PASSWORD`; never commit credentials to GitHub.

The trained model stays on the Linux laptop. A phone acts only as a wireless
microphone and browser display: audio is sent to the laptop over WebSocket, the
laptop extracts the exact training features and loads
`models/isolation_forest.joblib`, then the prediction is sent back to the phone.
The phone does not train or run the model locally.

## Installation and dashboard

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m dashboard
```

The primary dashboard is now FastHTML + MonsterUI at `http://localhost:5001`.
It reads the same trained model and SQLite events as the monitoring services.
The earlier Streamlit dashboard remains available with `streamlit run app.py`
for legacy inspection, but new UI work should target `dashboard.py`.

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

### Smartphone microphone monitoring

Install the added FastAPI dependencies and start the phone server on the laptop:

```bash
python -m pip install -r requirements.txt
python -m modules.phone_stream
```

On the same Wi-Fi, open `http://LAPTOP_IP:8000` on the phone. On a remote
network, use an HTTPS tunnel or reverse proxy; browser microphone permissions
normally require a secure origin. Grant microphone permission, enter a machine
ID, and press **Start monitoring**. The phone sends PCM16 audio over WebSocket;
the laptop decodes and resamples it before using the existing feature extractor
and trained model. The Streamlit **Phone Monitoring** page shows the latest
result and phone events.

The phone page reports the final result when monitoring stops as `NORMAL` or
`ANOMALY`, together with RMS, peak amplitude, dominant frequency, and anomaly
score. These are real values from the last processed audio window; no result is
shown until the laptop has successfully processed at least one window.

For a quick local check, open the page on the laptop first and use the laptop
IP from the phone. Do not expose port 8000 publicly without authentication,
HTTPS, and a firewall rule limiting access.

### Phone demo script

After installing dependencies, run the included helper:

```bash
chmod +x scripts/demo_phone_monitor.sh
./scripts/demo_phone_monitor.sh
```

In a second terminal, expose the phone server securely:

```bash
ngrok http 8000
```

Open the generated `https://...` URL on the phone, grant microphone access,
enter a machine ID, and press **Start monitoring**. Wait for the connection and
first model result, then press **Stop monitoring** to see the final label. The
Streamlit dashboard can be run separately with:

```bash
.venv/bin/streamlit run app.py
```

The phone events will appear in the dashboard's **Phone Monitoring** and
**Alerts & History** pages. Re-train the laptop model with representative
healthy recordings before treating results as meaningful.

The phone server and dashboard run at the same time as separate processes:

```bash
# Terminal 1: phone audio backend
.venv/bin/python -m modules.phone_stream

# Terminal 2: Streamlit dashboard
.venv/bin/streamlit run app.py

# Terminal 3: HTTPS tunnel for the phone
ngrok http 8000
```

The dashboard reads the phone session state and SQLite events written by the
FastAPI process. The **Phone Monitoring** page refreshes automatically while it
is open, so the latest phone result appears there without a manual reload.

### Community

The **Community** page stores questions, answers, and likes in
`data/kairos_community.db` using SQLite, so posts survive Streamlit restarts.
Questions are sorted by helpful votes and then newest-first; the database
primary key on `(question_id, user_id)` prevents duplicate likes. The entire
KAIROS-AI dashboard requires Google Sign-In through Firebase Authentication;
the browser
uses Firebase's official Web SDK, while the Python backend verifies every ID
token with Firebase Admin before accepting a question, answer, like, unlike,
or deletion. Service-account credentials are never sent to the browser.
Content is redacted by the reusable `modules/community_filter.py` module
before it is saved.

#### Firebase setup

1. Create or select a Firebase project at the Firebase Console.
2. Add a Web app and copy its Web configuration values.
3. In **Authentication → Sign-in method**, enable the Google provider and add
   the app's local/deployed domain to the authorized domains.
4. In **Project settings → Service accounts**, create a service-account key
   for the backend. Store it only in protected deployment secrets.

For local development, configure the Web values as environment variables:

```text
FIREBASE_API_KEY=...
FIREBASE_AUTH_DOMAIN=your-project.firebaseapp.com
FIREBASE_PROJECT_ID=your-project-id
FIREBASE_STORAGE_BUCKET=...
FIREBASE_MESSAGING_SENDER_ID=...
FIREBASE_APP_ID=...
# Recommended locally: use the downloaded service-account JSON file
FIREBASE_ADMIN_CREDENTIALS_FILE=/absolute/path/to/firebase-adminsdk.json
# Alternative for secret managers:
# FIREBASE_ADMIN_CREDENTIALS_JSON={"type":"service_account", ...}
```

The application reads these values from protected environment configuration
only. Copy `.env.example` to a local, ignored `.env` file and populate it, or
configure the same variables in the deployment environment. Never commit the
service-account JSON, private key, or passwords. The Community API is mounted
at `/api/community` and verifies the Firebase bearer token on every request;
it does not fall back to anonymous/session identities when configuration is
missing.

Existing SQLite rows are preserved without migration. Questions created by the
older prototype may contain identities that cannot be verified as Firebase
UIDs; they remain readable, but no new Firebase user can delete them unless
the verified UID already matches the stored owner. No legacy question is
assigned to an arbitrary Firebase user.

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
