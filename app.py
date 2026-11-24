"""KAIROS-AI — Acoustic Intelligence for Predictive Machine Health Monitoring."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

from modules.anomaly_detection import MODEL_PATH, load_model, model_status, predict_anomaly, train_from_directory
from modules.alerts import EventStore
from modules.audio_input import load_audio, record_audio
from modules.feature_extraction import FeatureExtractionError, extract_features_from_audio
from modules.machine_manager import get_machines, machine_snapshot, utc_timestamp
from modules.signal_processing import ScientificDependencyError, calculate_fft, generate_spectrogram, generate_waveform


st.set_page_config(page_title="KAIROS-AI", page_icon="◈", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
:root { --navy:#07111f; --panel:#0d1b2d; --line:#20354d; --cyan:#43d9e8; --green:#48d597; --amber:#f2b84b; --red:#ff647c; }
.stApp { background:var(--navy); color:#e8f1f7; }
[data-testid="stSidebar"] { background:#091827; border-right:1px solid var(--line); }
.brand { letter-spacing:.16em; font-size:.82rem; color:var(--cyan); font-weight:700; }
.tagline { color:#8ca5b8; font-size:.9rem; margin-bottom:1.4rem; }
.panel { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:1rem 1.15rem; margin-bottom:1rem; }
.metric-label { color:#8ca5b8; font-size:.78rem; text-transform:uppercase; letter-spacing:.08em; }
.metric-value { font-size:1.7rem; font-weight:700; margin-top:.25rem; }
.status { display:inline-block; padding:.25rem .55rem; border-radius:20px; font-size:.72rem; font-weight:700; letter-spacing:.05em; }
.healthy { color:var(--green); background:#10382e; } .warning { color:var(--amber); background:#3b2d12; } .critical { color:var(--red); background:#431c28; } .neutral { color:#a9bdcc; background:#1a2b3d; }
.small { color:#8ca5b8; font-size:.82rem; }
</style>
""", unsafe_allow_html=True)


def init_state() -> None:
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("analysis", None)
    st.session_state.setdefault("selected_machine", "MTR-01")
    st.session_state.setdefault("live_process", None)


def status_badge(status: str) -> str:
    cls = "healthy" if status == "NORMAL" else "critical" if status == "ANOMALY" else "neutral"
    return f'<span class="status {cls}">{status}</span>'


def chart_layout(fig, title: str):
    fig.update_layout(title=title, template="plotly_dark", paper_bgcolor="#0d1b2d", plot_bgcolor="#0d1b2d", margin=dict(l=45, r=20, t=45, b=40), height=300, font=dict(color="#d9e8f0"))
    return fig


def audio_bytes(audio, sample_rate: int) -> bytes:
    """Encode in-memory samples as WAV for reliable browser preview."""
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("Audio preview is unavailable because SoundFile could not be imported.") from exc
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    return buffer.getvalue()


def live_events(limit: int = 500) -> list[dict]:
    event_db = Path("data/kairos_events.db")
    if not event_db.is_file():
        return []
    store = EventStore(event_db)
    try:
        return store.recent(limit)
    finally:
        store.close()


def overview() -> None:
    st.title("Overview")
    st.caption("Acoustic Intelligence & Predictive Machine Health Monitoring")
    events = live_events()
    session_records = st.session_state.history
    anomalies = [event for event in events if event["status"] == "ANOMALY"]
    latest = events[0] if events else None
    model = model_status()
    cols = st.columns(4)
    for col, label, value in zip(cols, ["Live events", "Live anomalies", "Emails sent", "Session analyses"], [len(events), len(anomalies), sum(bool(event["alert_sent"]) for event in events), len(session_records)]):
        with col:
            st.markdown(f'<div class="panel"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)
    model_text = model["detail"]
    latest_text = f"Latest live result: {latest['status']} for {latest['machine_id']} at {latest['timestamp']}" if latest else "No live monitoring events recorded yet."
    st.markdown(f'<div class="panel"><b>System status</b> &nbsp; {status_badge("TRAINED" if model["status"] == "TRAINED" else model["status"])}<div class="small">{model_text}<br>{latest_text}</div></div>', unsafe_allow_html=True)
    if events:
        st.subheader("Latest real events")
        st.dataframe(events[:10], use_container_width=True, hide_index=True)
    else:
        st.info("No live data yet. Start Live Monitoring or analyze an audio recording.")


def acoustic_analysis() -> None:
    st.title("Acoustic Analysis")
    machines = get_machines()
    labels = {m["name"]: m["id"] for m in machines}
    selected = st.selectbox("Machine", list(labels), index=[m["id"] for m in machines].index(st.session_state.selected_machine))
    st.session_state.selected_machine = labels[selected]
    source = st.radio("Audio source", ["Upload recording", "Laptop microphone"], horizontal=True)
    audio = None
    rate = None
    if source == "Upload recording":
        upload = st.file_uploader("Upload WAV, FLAC, MP3, or OGG", type=["wav", "flac", "mp3", "ogg"])
        if upload is not None:
            st.audio(upload.getvalue())
            if st.button("Load recording"):
                try:
                    audio, rate = load_audio(io.BytesIO(upload.getvalue()))
                    st.session_state.pending_audio = (audio, rate)
                    st.success(f"Loaded {len(audio) / rate:.1f} seconds at {rate:,} Hz.")
                except ValueError as exc:
                    st.error(str(exc))
    else:
        duration = st.slider("Recording duration (seconds)", 1, 15, 5)
        rate_choice = st.selectbox("Sampling rate", [16000, 22050, 44100], index=0)
        if st.button("Record from microphone"):
            try:
                st.session_state.pending_audio = record_audio(duration, rate_choice)
                st.success("Recording captured and ready for analysis.")
            except RuntimeError as exc:
                st.error(str(exc))
    if "pending_audio" in st.session_state:
        audio, rate = st.session_state.pending_audio
        st.audio(audio_bytes(audio, rate), format="audio/wav")
        if st.button("ANALYZE AUDIO", type="primary"):
            with st.spinner("Extracting acoustic features and running diagnostics…"):
                try:
                    features = extract_features_from_audio(audio, rate)
                    model = load_model() if model_status()["status"] == "TRAINED" else None
                    result = predict_anomaly(features, model)
                    st.session_state.analysis = {"audio": audio, "rate": rate, "features": features, "result": result, "machine_id": st.session_state.selected_machine}
                    st.session_state.history.append({"machine_id": st.session_state.selected_machine, "timestamp": utc_timestamp(), "result": result["status"], "score": result["score"]})
                except (ScientificDependencyError, FeatureExtractionError) as exc:
                    st.warning(str(exc))
                    st.info("Waveform and FFT remain available. MFCC/spectral features and AI diagnostics are temporarily unavailable.")
    render_analysis(st.session_state.analysis)


def render_analysis(analysis: dict | None) -> None:
    if not analysis:
        st.info("Load or record audio, then choose Analyze Audio to see acoustic intelligence results.")
        return
    audio, rate, features, result = analysis["audio"], analysis["rate"], analysis["features"], analysis["result"]
    st.subheader("Signal views")
    time, amplitude = generate_waveform(audio, rate)
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.warning("Interactive charts are temporarily unavailable because Plotly could not be imported.")
        st.write({"waveform_samples": len(amplitude), "duration_seconds": round(time[-1], 3) if time else 0})
        return
    fig = go.Figure(go.Scatter(x=time, y=amplitude, mode="lines", line=dict(color="#43d9e8", width=1)))
    st.plotly_chart(chart_layout(fig, "Waveform · amplitude over time"), use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        freq, magnitude, dominant = calculate_fft(audio, rate)
        fft = go.Figure(go.Scatter(x=freq, y=magnitude, mode="lines", line=dict(color="#43d9e8", width=1)))
        fft.update_xaxes(title="Frequency (Hz)"); fft.update_yaxes(title="Magnitude")
        st.plotly_chart(chart_layout(fft, f"FFT frequency spectrum · dominant {dominant:.1f} Hz"), use_container_width=True)
    with c2:
        try:
            db, times, freqs = generate_spectrogram(audio, rate)
            spec = go.Figure(go.Heatmap(z=db, x=times, y=freqs, colorscale="Viridis", colorbar=dict(title="dB")))
            spec.update_xaxes(title="Time (s)"); spec.update_yaxes(title="Frequency (Hz)")
            st.plotly_chart(chart_layout(spec, "Spectrogram"), use_container_width=True)
        except ScientificDependencyError as exc:
            st.warning(f"Spectrogram unavailable: {exc}")
    st.subheader("Diagnostic output")
    left, right = st.columns([1, 2])
    with left:
        st.markdown(f'<div class="panel"><div class="metric-label">Model result</div><div class="metric-value">{status_badge(result["status"])}</div><br><span class="small">Relative score: {result["score"] if result["score"] is not None else "—"}</span></div>', unsafe_allow_html=True)
    with right:
        st.markdown(f'<div class="panel"><b>Interpretation</b><br>{result["explanation"]}</div>', unsafe_allow_html=True)
    st.table([{"Feature": key, "Value": value} for key, value in features.items()])


def diagnostics() -> None:
    st.title("AI Diagnostics")
    current_model_status = model_status()
    st.markdown(f'<div class="panel"><div class="metric-label">Model status</div><div class="metric-value">{status_badge(current_model_status["status"])}</div><div class="small">Algorithm: IsolationForest · saved model: {MODEL_PATH.name if current_model_status["status"] == "TRAINED" else "none"}</div></div>', unsafe_allow_html=True)
    st.write("Train only with representative healthy recordings. The model reports relative acoustic similarity, not a failure probability.")
    if st.button("Train from dataset/healthy"):
        try:
            summary = train_from_directory(__import__("pathlib").Path("dataset"))
            st.success(f"Model trained from {summary['samples']} healthy recordings.")
        except (ValueError, ScientificDependencyError) as exc:
            st.warning(str(exc))
    if st.session_state.analysis:
        result = st.session_state.analysis["result"]
        st.subheader("Latest result")
        st.markdown(status_badge(result["status"]), unsafe_allow_html=True)
        st.info("Potential acoustic anomaly — further inspection recommended." if result["status"] == "ANOMALY" else result["explanation"])
    else:
        st.info("Run an audio analysis to populate diagnostics.")


def monitoring() -> None:
    st.title("Machine Monitoring")
    st.caption("Machine status from actual live-monitoring events")
    events = live_events()
    if not events:
        st.info("No machine events recorded yet. Start Live Monitoring to populate this page.")
        return
    latest_by_machine = {}
    for event in events:
        latest_by_machine.setdefault(event["machine_id"], event)
    for machine_id, event in latest_by_machine.items():
        st.markdown(f'<div class="panel"><b>{machine_id}</b> &nbsp; {status_badge(str(event["status"]))}<br><span class="small">Last event: {event["timestamp"]} · anomaly score: {event["anomaly_score"]:.6f}</span></div>', unsafe_allow_html=True)


def live_monitoring() -> None:
    st.title("Live Monitoring")
    st.caption("Continuous microphone analysis with optional email alerts")
    process = st.session_state.live_process
    if process is not None and process.poll() is not None:
        log_path = Path("data/live_monitor.log")
        if log_path.is_file():
            try:
                log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
            except OSError:
                log_tail = ""
            if log_tail:
                st.error("Live monitoring stopped. Last worker output:")
                st.code(log_tail)
        st.session_state.live_process = None
        process = None

    if process is not None:
        st.success(f"Live monitoring is running (process {process.pid}).")
        _render_live_waveform()
        if st.button("Stop live monitoring", type="primary"):
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            st.session_state.live_process = None
            st.rerun()
    else:
        st.info("Monitoring is stopped. Start it to analyze the default microphone continuously.")
        machines = get_machines()
        labels = {machine["name"]: machine["id"] for machine in machines}
        selected_name = st.selectbox("Machine", list(labels), key="live_machine")
        device_options: dict[str, int] = {}
        try:
            import sounddevice as sd
            for index, device in enumerate(sd.query_devices()):
                if int(device.get("max_input_channels", 0)) > 0:
                    label = f"{index}: {device['name']} · {int(device['max_input_channels'])} input channel(s)"
                    device_options[label] = index
        except Exception as exc:
            st.error(f"Could not list microphone devices: {exc}")
        if not device_options:
            st.error("No microphone input devices were found.")
            return
        selected_device_label = st.selectbox("Microphone input", list(device_options), key="live_device")
        selected_device = device_options[selected_device_label]
        st.caption("Use the same microphone and placement used to record the normal training examples.")
        window_seconds = st.number_input("Analysis window (seconds)", min_value=2.0, max_value=60.0, value=5.0, step=1.0)
        interval_seconds = st.number_input("Analysis interval (seconds)", min_value=1.0, max_value=60.0, value=5.0, step=1.0)
        required_anomalies = st.number_input("Anomalous windows before alert", min_value=1, max_value=20, value=3, step=1)
        cooldown_minutes = st.number_input("Email cooldown (minutes)", min_value=0, max_value=1440, value=60, step=5)
        send_email = st.checkbox("Send email alerts", value=False)
        email_values: dict[str, str] = {}
        if send_email:
            st.warning("SMTP credentials are used only by the monitor process and are not saved to the repository.")
            email_values = {
                "SMTP_HOST": st.text_input("SMTP host"),
                "SMTP_PORT": st.text_input("SMTP port", value="587"),
                "SMTP_USERNAME": st.text_input("SMTP username"),
                "SMTP_PASSWORD": st.text_input("SMTP password", type="password"),
                "ALERT_FROM": st.text_input("From address"),
                "ALERT_TO": st.text_input("To address"),
            }
        if st.button("Start live monitoring", type="primary"):
            status = model_status()
            if status["status"] != "TRAINED":
                st.error("Train the model before starting live monitoring.")
            elif send_email and any(not value for value in email_values.values()):
                st.error("Complete all email settings or turn off email alerts.")
            else:
                environment = os.environ.copy()
                if send_email:
                    environment.update(email_values)
                command = [
                    sys.executable, "-m", "modules.live_monitor",
                    "--machine-id", labels[selected_name],
                    "--device", str(selected_device),
                    "--window-seconds", str(window_seconds),
                    "--interval-seconds", str(interval_seconds),
                    "--required-anomalies", str(required_anomalies),
                    "--cooldown-seconds", str(int(cooldown_minutes * 60)),
                ]
                if not send_email:
                    command.append("--no-email")
                log_path = Path("data/live_monitor.log")
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_handle = log_path.open("w", encoding="utf-8")
                st.session_state.live_process = subprocess.Popen(
                    command, cwd=Path(__file__).resolve().parent, env=environment,
                    stdout=log_handle, stderr=subprocess.STDOUT,
                )
                log_handle.close()
                st.rerun()

    st.subheader("Recent live events")
    if st.button("Clear live history", type="secondary"):
        _confirm_clear_live_history()
    event_db = Path("data/kairos_events.db")
    if event_db.is_file():
        store = EventStore(event_db)
        try:
            events = store.recent(20)
        finally:
            store.close()
        if events:
            st.caption("Legend: timestamp = event time (UTC) · machine_id = monitored machine · status = model result · anomaly_score = higher means more unusual · decision_score = model decision value · alert_sent = whether an email was sent · message = readable interpretation")
            anomaly_count = sum(event["status"] == "ANOMALY" for event in events)
            alert_count = sum(bool(event["alert_sent"]) for event in events)
            metric_columns = st.columns(3)
            metric_columns[0].metric("Events shown", len(events))
            metric_columns[1].metric("Anomalies", anomaly_count)
            metric_columns[2].metric("Emails sent", alert_count)
            try:
                import pandas as pd
                import plotly.express as px
                chart_data = pd.DataFrame(events).sort_values("timestamp")
                st.plotly_chart(
                    px.line(chart_data, x="timestamp", y="anomaly_score", color="machine_id", markers=True,
                            title="Recent anomaly scores"), use_container_width=True,
                )
                status_data = pd.DataFrame({"status": ["NORMAL", "ANOMALY"], "count": [len(events) - anomaly_count, anomaly_count]})
                st.plotly_chart(
                    px.bar(status_data, x="status", y="count", color="status", title="Recent result counts"),
                    use_container_width=True,
                )
            except ImportError:
                st.info("Charts require Pandas and Plotly; the event table is still available.")
            st.dataframe(events, use_container_width=True, hide_index=True)
        else:
            st.info("No live events recorded yet.")
    else:
        st.info("No event database exists yet. Start monitoring to create it.")


def phone_monitoring() -> None:
    st.title("Phone Live Monitoring")
    st.caption("Smartphone microphone → WebSocket → laptop model")
    st.code(".venv/bin/python -m modules.phone_stream", language="bash")
    st.write("Start the phone audio server, then open its address on the phone. The trained model remains on this laptop.")
    state_path = Path("data/phone_live_state.json")
    if not state_path.is_file():
        st.info("Phone server has not reported a session yet.")
        return
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        st.warning("Phone session state is temporarily unavailable.")
        return
    if not state.get("connected"):
        st.warning("Phone is disconnected.")
    else:
        st.success(f"Phone connected for {state.get('machine_id', 'unknown machine')}.")
    if state.get("status"):
        st.metric("Latest result", state["status"])
        cols = st.columns(4)
        cols[0].metric("RMS", f"{state.get('rms', 0):.4f}")
        cols[1].metric("Peak", f"{state.get('peak', 0):.4f}")
        cols[2].metric("Dominant frequency", f"{state.get('dominant_frequency', 0):.1f} Hz")
        cols[3].metric("Anomaly score", f"{state.get('anomaly_score', 0):.4f}")
    events = [event for event in live_events() if str(event.get("machine_id", "")).startswith("PHONE")]
    if events:
        st.subheader("Phone events")
        st.dataframe(events[:20], use_container_width=True, hide_index=True)


@st.dialog("Clear live monitoring history")
def _confirm_clear_live_history() -> None:
    st.warning("This will permanently remove the recorded live events and reset the waveform snapshot. The trained model will not be affected.")
    confirm = st.checkbox("I understand that the live history will be deleted.")
    left, right = st.columns(2)
    with left:
        if st.button("Cancel"):
            st.rerun()
    with right:
        if st.button("Delete history", type="primary", disabled=not confirm):
            event_db = Path("data/kairos_events.db")
            if event_db.is_file():
                store = EventStore(event_db)
                try:
                    # Keep the UI compatible with a monitor process that imported
                    # an older EventStore module during Streamlit hot reload.
                    if hasattr(store, "clear"):
                        store.clear()
                    else:
                        store.connection.execute("DELETE FROM events")
                        store.connection.commit()
                finally:
                    store.close()
            waveform_path = Path("data/live_waveform.json")
            if waveform_path.is_file():
                waveform_path.unlink()
            st.success("Live history cleared.")
            st.rerun()


@st.fragment(run_every="1s")
def _render_live_waveform() -> None:
    st.subheader("Live audio signal")
    waveform_path = Path("data/live_waveform.json")
    if not waveform_path.is_file():
        st.info("Waiting for the first audio window…")
        return
    try:
        payload = json.loads(waveform_path.read_text(encoding="utf-8"))
        samples = payload.get("samples", [])
        if not samples:
            st.info("Waiting for the first audio window…")
            return
        st.line_chart(samples, height=220)
        st.caption("The chart refreshes while the monitor is running and shows the latest analysis window.")
    except (OSError, ValueError, TypeError):
        st.info("Waiting for a readable waveform snapshot…")


def alerts() -> None:
    st.title("Alerts & History")
    if st.button("Clear session history"):
        st.session_state.history = []
        st.rerun()
    events = live_events()
    if not st.session_state.history and not events:
        st.info("No alerts or analyses recorded yet.")
        return
    if events:
        st.subheader("Live monitoring history")
        st.dataframe(events, use_container_width=True, hide_index=True)
    if st.session_state.history:
        st.subheader("Current dashboard session")
        st.table(st.session_state.history)


def about() -> None:
    st.title("About KAIROS-AI")
    st.markdown("**Every machine has a voice. We detect when something goes wrong.**")
    st.write("KAIROS-AI combines signal processing and anomaly detection for non-invasive machine health monitoring. It is designed for team development: audio input, signal processing, model logic, machine registry, and presentation are separated into modules.")
    st.info("The dashboard displays model results from uploaded/recorded audio and live-monitoring events. It does not fabricate sensor readings or claim fault probabilities.")


init_state()
with st.sidebar:
    st.markdown('<div class="brand">KAIROS-AI</div><div class="tagline">Acoustic Intelligence for Predictive Machine Health Monitoring</div>', unsafe_allow_html=True)
    page = st.radio("Navigate", ["Overview", "Acoustic Analysis", "AI Diagnostics", "Live Monitoring", "Phone Monitoring", "Machine Monitoring", "Alerts & History", "About Project"], label_visibility="collapsed")
    st.divider()
    st.caption("Prototype mode · audio-first")

{"Overview": overview, "Acoustic Analysis": acoustic_analysis, "AI Diagnostics": diagnostics, "Live Monitoring": live_monitoring, "Phone Monitoring": phone_monitoring, "Machine Monitoring": monitoring, "Alerts & History": alerts, "About Project": about}[page]()
