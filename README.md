# KAIROS-AI

Acoustic Intelligence for Predictive Machine Health Monitoring.

KAIROS-AI analyzes machine sound and presents signal views, anomaly scores,
live monitoring, alerts, and a Firebase-authenticated discussion community.
It is a development platform: `NORMAL` and `ANOMALY` are model classifications,
not exact fault identification or a guaranteed failure probability.

## Features

- WAV/FLAC/MP3/OGG upload and laptop microphone recording.
- Phone microphone streaming over WebSocket.
- Waveform, FFT, spectrogram, MFCC, and spectral feature analysis.
- Healthy-sound training with StandardScaler + Isolation Forest.
- `NORMAL` / `ANOMALY` classification and AI diagnostics.
- Machine monitoring, event history, and optional SMTP email alerts.
- Firebase Google Authentication with server-side Firebase Admin token checks.
- Community questions, answers, profanity redaction, persistent likes, and
  ownership-protected deletion.
- Responsive FastHTML + MonsterUI HTML dashboard.
- Legacy Streamlit dashboard retained for compatibility.

## Architecture

```text
dashboard.py                  FastHTML + MonsterUI dashboard, port 5001
app.py                        Legacy Streamlit dashboard, port 8501
modules/firebase_core.py      Shared Firebase config and Admin verification
modules/firebase_auth.py      Streamlit Firebase authentication integration
modules/community_api.py      Authenticated FastAPI Community API
modules/community_store.py    SQLite questions, answers, and likes
modules/phone_stream.py       FastAPI/WebSocket phone microphone server
modules/audio_input.py        Audio loading and recording
modules/signal_processing.py  Waveform, FFT, spectrogram helpers
modules/feature_extraction.py Fixed feature schema including MFCC features
modules/anomaly_detection.py  Isolation Forest model loading/inference
modules/live_monitor.py       Continuous laptop microphone worker
modules/alerts.py             SQLite event storage and SMTP notifications
modules/train.py              Healthy-recording model training
modules/predict.py             Command-line prediction
modules/evaluation.py          Evaluation reports and plots
web/firebase_auth/             Streamlit Firebase component frontend
tests/                         Unit and API tests
data/                          Local SQLite/state files, ignored by Git
```

## Windows installation

Use a supported 64-bit Python installation and PowerShell:

```powershell
cd D:\kairos-ai
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use the interpreter directly:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Linux installation

On Ubuntu/Debian, install the system packages commonly needed for audio
decoding, microphone capture, and scientific Python wheels:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip build-essential \
  libsndfile1 portaudio19-dev
```

Create and activate the project virtual environment:

```bash
cd /path/to/kairos-ai
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Copy and edit the environment template before starting an authenticated
dashboard:

```bash
cp .env.example .env
chmod 600 .env
$EDITOR .env
```

Keep Firebase Admin credentials outside the repository when possible. For
example, configure a protected service-account file path in `.env`:

```text
FIREBASE_ADMIN_CREDENTIALS_FILE=/etc/kairos/firebase-adminsdk.json
```

The file should be readable only by the account running KAIROS-AI:

```bash
chmod 600 /etc/kairos/firebase-adminsdk.json
```

## Starting on Linux

With `.venv` activated, run the primary FastHTML dashboard:

```bash
python dashboard.py
```

Open <http://localhost:5001>. The equivalent module command is:

```bash
python -m dashboard
```

Run the legacy Streamlit dashboard with:

```bash
streamlit run app.py
```

It is available at <http://localhost:8501>. Run the phone microphone server in
a separate terminal, with the virtual environment activated:

```bash
python -m modules.phone_stream
```

The phone server listens on `KAIROS_PHONE_PORT` or port `8000`. For a remote
phone, use an HTTPS reverse proxy or tunnel because browser microphone access
normally requires a secure origin. The Community API is already mounted inside
the FastHTML dashboard at `/api/community`; it does not require a second
process.

For a long-running Linux deployment, use a process supervisor such as systemd,
Supervisor, or Docker rather than leaving a shell session open. Set the working
directory to the repository root so the local `data/`, `models/`, and `.env`
paths resolve consistently. Bind behind an HTTPS reverse proxy before exposing
the dashboard outside localhost.

## Configuration

Copy `.env.example` to `.env`. `.env` is ignored by Git and must never be
committed. Fill in the Firebase Web values from the Firebase Web app settings:

```text
FIREBASE_API_KEY=
FIREBASE_AUTH_DOMAIN=
FIREBASE_PROJECT_ID=
FIREBASE_STORAGE_BUCKET=
FIREBASE_MESSAGING_SENDER_ID=
FIREBASE_APP_ID=
```

In Firebase Console, enable Google under Authentication → Sign-in method and
add local/deployed domains under authorized domains. For server verification,
create a Firebase Admin service-account key and configure one of these protected
options:

```text
FIREBASE_ADMIN_CREDENTIALS_FILE=C:\secure\firebase-adminsdk.json
```

or:

```text
FIREBASE_ADMIN_CREDENTIALS_JSON={"type":"service_account", ...}
```

The service-account JSON and private key must never be committed to GitHub or
sent to the browser. If Firebase configuration is missing, the dashboards show
a setup error instead of falling back to anonymous identities.

SMTP alerts use these environment variables when alerts are enabled:

```text
SMTP_HOST=...
SMTP_PORT=587
SMTP_USERNAME=...
SMTP_PASSWORD=...
ALERT_FROM=...
ALERT_TO=person1@example.com,person2@example.com
SMTP_USE_TLS=true
```

`ALERT_TO` supports up to three comma- or semicolon-separated recipients.

## Starting the applications

### FastHTML dashboard (primary)

```powershell
.venv\Scripts\Activate.ps1
python dashboard.py
```

Open <http://localhost:5001>. The equivalent module command is:

```powershell
python -m dashboard
```

Do not use `python -m dashboard.py`; `-m` expects a module name without the
`.py` suffix. On Windows, the dashboard disables FastHTML's development
reloader by default to avoid an extra-process startup path. Opt in with:

```powershell
$env:KAIROS_RELOAD="true"
python dashboard.py
```

### Legacy Streamlit dashboard

```powershell
.venv\Scripts\streamlit.exe run app.py
```

Open <http://localhost:8501>. The primary HTML dashboard should be used for new
UI work; Streamlit remains supported for the existing audio-first interface.

### Phone microphone server

Run separately from a dashboard:

```powershell
.venv\Scripts\python.exe -m modules.phone_stream
```

It listens on `KAIROS_PHONE_PORT` or port `8000`. Open the displayed phone page
on the same network. Browser microphone access normally requires HTTPS when
used outside localhost; use an HTTPS tunnel or reverse proxy and do not expose
the unauthenticated phone endpoint publicly.

### Community API

The Community API is mounted inside `dashboard.py` at `/api/community`. It is
not a separate process. Every request requires a Firebase Bearer ID token and
is verified by Firebase Admin before the API reads or writes community data.

## Data and persistence

SQLite files under `data/` store monitoring events and Community data locally.
The Community database preserves questions, answers, and likes across restarts;
the `(question_id, user_id)` primary key prevents duplicate likes. Existing
legacy records are preserved. Records whose old owner cannot be verified as a
Firebase UID remain readable but cannot be assigned to an arbitrary user.

SQLite is suitable for a single local deployment. Separate machines or cloud
deployments do not share data unless the database is deliberately moved or
replaced with a shared database service. Do not place a live SQLite file on an
unreliable network share.

## Windows lxml/Application Control issue

MonsterUI's `monsterui.all` imports `monsterui.franken`, and that module imports
compiled `lxml.html`/`lxml.etree` at import time. On some Windows PCs,
Application Control blocks the compiled lxml DLL before the dashboard starts.

The dashboard only used MonsterUI's `Theme`, `Card`, and `UkIcon` helpers. It now
imports `Theme` from the lxml-free `monsterui.core` module and uses small local
equivalents for `Card` and `UkIcon`. This preserves the existing HTML dashboard
layout and avoids importing `monsterui.franken` during startup. The MonsterUI
package may still install lxml as a package dependency, but KAIROS-AI does not
load it for dashboard startup. Application Control is not bypassed or weakened.

If a fresh installation itself is blocked by your organization's policy, ask
your administrator to approve the signed dependency through the normal policy
process. Do not disable Application Control or randomly downgrade Python.

## Tests and verification

Run the complete suite from the project virtual environment:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe -m py_compile dashboard.py app.py
Get-ChildItem modules, tests -Filter *.py -Recurse | ForEach-Object { .venv\Scripts\python.exe -m py_compile $_.FullName }
git diff --check
```

The tests cover feature extraction, anomaly behavior, Firebase token handling,
Community API authentication, persistence, likes, profanity filtering, and
ownership protection.

## Future scope

ESP32 + INMP441 hardware acquisition, authenticated device provisioning,
multi-machine fleet storage, and production-scale shared database deployment
are future work. They are not represented as completed features.
