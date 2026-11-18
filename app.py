"""KAIROS-AI — Acoustic Intelligence for Predictive Machine Health Monitoring."""

from __future__ import annotations

import io
from datetime import datetime

import streamlit as st

from modules.anomaly_detection import MODEL_PATH, load_model, model_status, predict_anomaly, train_from_directory
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


def overview() -> None:
    st.title("Overview")
    st.caption("Acoustic Intelligence & Predictive Maintenance · Demonstration workspace")
    snapshot = machine_snapshot(st.session_state.history)
    anomalies = [m for m in snapshot if m["status"] == "ANOMALY"]
    cols = st.columns(4)
    for col, label, value in zip(cols, ["Machines monitored", "Healthy machines", "Anomalies detected", "Audio analyses"], [len(snapshot), len(snapshot) - len(anomalies), len(anomalies), len(st.session_state.history)]):
        with col:
            st.markdown(f'<div class="panel"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="panel"><b>System status</b> &nbsp; <span class="status healthy">OPERATIONAL</span><div class="small">No live IoT feed is connected. Values below are demonstration state until audio is analyzed.</div></div>', unsafe_allow_html=True)
    st.subheader("Machine fleet")
    cards = st.columns(4)
    for col, machine in zip(cards, snapshot):
        with col:
            st.markdown(f'<div class="panel"><b>{machine["name"]}</b><br><span class="small">{machine["id"]} · {machine["type"]}</span><br><br>{status_badge(str(machine["status"]))}<br><br><span class="small">Last analysis: {machine["last_analysis"]}</span></div>', unsafe_allow_html=True)


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
    st.caption("Demonstration fleet registry · live ESP32 telemetry is not connected")
    for machine in machine_snapshot(st.session_state.history):
        st.markdown(f'<div class="panel"><b>{machine["name"]}</b> &nbsp; {status_badge(str(machine["status"]))}<br><span class="small">ID {machine["id"]} · {machine["type"]} · Last analyzed {machine["last_analysis"]}</span></div>', unsafe_allow_html=True)
    st.subheader("Future hardware integration")
    st.write("An ESP32 DevKit with an INMP441 I2S microphone can eventually publish audio windows and metadata over MQTT or HTTP. An ingestion adapter would feed those windows into the existing preprocessing pipeline; this prototype intentionally does not fabricate live IoT data.")


def alerts() -> None:
    st.title("Alerts & History")
    if st.button("Clear session history"):
        st.session_state.history = []
        st.rerun()
    if not st.session_state.history:
        st.info("No alerts or analyses in this session.")
        return
    st.table(st.session_state.history)


def about() -> None:
    st.title("About KAIROS-AI")
    st.markdown("**Every machine has a voice. We detect when something goes wrong.**")
    st.write("KAIROS-AI combines signal processing and anomaly detection for non-invasive machine health monitoring. It is designed for team development: audio input, signal processing, model logic, machine registry, and presentation are separated into modules.")
    st.info("Prototype boundary: no labelled fault classifier, no claimed accuracy, no live hardware feed, and no fake sensor history.")


init_state()
with st.sidebar:
    st.markdown('<div class="brand">KAIROS-AI</div><div class="tagline">Acoustic Intelligence for Predictive Machine Health Monitoring</div>', unsafe_allow_html=True)
    page = st.radio("Navigate", ["Overview", "Acoustic Analysis", "AI Diagnostics", "Machine Monitoring", "Alerts & History", "About Project"])
    st.divider()
    st.caption("Prototype mode · audio-first")

{"Overview": overview, "Acoustic Analysis": acoustic_analysis, "AI Diagnostics": diagnostics, "Machine Monitoring": monitoring, "Alerts & History": alerts, "About Project": about}[page]()
