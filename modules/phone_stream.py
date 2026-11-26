"""FastAPI/WebSocket bridge for smartphone microphone monitoring."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from .alerts import AlertPolicy, EventStore
from .anomaly_detection import DEFAULT_MODEL_PATH, load_model, predict_from_features
from .feature_extraction import extract_features_from_audio

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "data" / "phone_live_state.json"


def decode_pcm16(payload: bytes, channels: int = 1) -> np.ndarray:
    """Decode little-endian signed PCM16 bytes into mono float samples."""
    if channels < 1:
        raise ValueError("channels must be positive")
    if len(payload) == 0 or len(payload) % (2 * channels) != 0:
        raise ValueError("PCM16 payload has an invalid byte length")
    samples = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples


class PhoneSession:
    def __init__(self, *, machine_id: str, sample_rate: int, window_seconds: float = 5.0) -> None:
        if sample_rate <= 0 or window_seconds <= 0:
            raise ValueError("sample_rate and window_seconds must be positive")
        self.machine_id = machine_id
        self.sample_rate = sample_rate
        self.window_seconds = window_seconds
        self.window_samples = int(sample_rate * window_seconds)
        self.buffer: deque[float] = deque(maxlen=self.window_samples)
        self.model = load_model()
        self.store = EventStore(ROOT / "data" / "kairos_events.db")
        self.policy = AlertPolicy()
        self.last_analysis = 0.0
        self.last_result: dict[str, Any] | None = None

    def close(self) -> None:
        self.store.close()

    def _write_state(self, **values: Any) -> None:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        state = {"updated_at": time.time(), "connected": True, **values}
        temporary = STATE_PATH.with_suffix(".tmp")
        temporary.write_text(json.dumps(state), encoding="utf-8")
        temporary.replace(STATE_PATH)

    def add_audio(self, samples: np.ndarray) -> dict[str, Any] | None:
        self.buffer.extend(samples.tolist())
        if len(self.buffer) < self.window_samples:
            self._write_state(buffer_seconds=len(self.buffer) / self.sample_rate)
            return None
        now = time.monotonic()
        if now - self.last_analysis < 5.0:
            return None
        self.last_analysis = now
        audio = np.asarray(self.buffer, dtype=np.float32)
        features = extract_features_from_audio(audio, self.sample_rate)
        result = predict_from_features(features, self.model)
        rms = float(np.sqrt(np.mean(np.square(audio))))
        peak = float(np.max(np.abs(audio)))
        spectrum = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
        dominant_frequency = float(np.argmax(spectrum) * self.sample_rate / len(audio)) if len(spectrum) else 0.0
        alert_sent = self.policy.observe(bool(result["is_anomaly"]), time.monotonic())
        self.store.record(
            machine_id=self.machine_id, status=str(result["status"]),
            anomaly_score=float(result["anomaly_score"]),
            decision_score=float(result["decision_score"]), alert_sent=alert_sent,
            message="Potential acoustic anomaly" if result["is_anomaly"] else "Normal",
        )
        self.last_result = {
            **result, "machine_id": self.machine_id, "rms": rms, "peak": peak,
            "dominant_frequency": dominant_frequency,
            "timestamp": time.time(),
        }
        self._write_state(**{**self.last_result, "buffer_seconds": self.window_seconds})
        return self.last_result


def create_app() -> Any:
    try:
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        from fastapi.responses import HTMLResponse
    except ImportError as exc:
        raise RuntimeError("Phone monitoring requires fastapi and uvicorn.") from exc

    app = FastAPI(title="KAIROS phone acoustic monitor")
    web_path = ROOT / "web" / "mobile_monitor.html"

    @app.get("/", response_class=HTMLResponse)
    async def mobile_page() -> str:
        return web_path.read_text(encoding="utf-8")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"ok": True, "model": str(DEFAULT_MODEL_PATH), "model_exists": DEFAULT_MODEL_PATH.is_file()}

    async def audio_socket(websocket: WebSocket) -> None:
        logger.info("WebSocket handshake received: path=%s headers=%s", websocket.url.path, dict(websocket.headers))
        await websocket.accept()
        session: PhoneSession | None = None
        try:
            hello = await asyncio.wait_for(websocket.receive_json(), timeout=10)
            sample_rate = int(hello.get("sample_rate", 48000))
            channels = int(hello.get("channels", 1))
            machine_id = str(hello.get("machine_id", "PHONE-01"))[:64]
            session = PhoneSession(machine_id=machine_id, sample_rate=sample_rate)
            await websocket.send_json({"type": "ready", "sample_rate": sample_rate})
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                payload = message.get("bytes")
                if payload is None:
                    continue
                if len(payload) > 1_000_000:
                    await websocket.send_json({"type": "error", "message": "Audio chunk too large"})
                    continue
                try:
                    samples = decode_pcm16(payload, channels)
                    result = await asyncio.to_thread(session.add_audio, samples)
                    if result:
                        await websocket.send_json({"type": "result", **result})
                    else:
                        await websocket.send_json({"type": "buffering"})
                except Exception as exc:
                    logger.exception("Phone audio chunk failed")
                    await websocket.send_json({"type": "error", "message": str(exc)})
        except WebSocketDisconnect:
            pass
        except Exception:
            logger.exception("Phone WebSocket failed")
        finally:
            if session is not None:
                session.close()
            try:
                STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
                STATE_PATH.write_text(json.dumps({"connected": False, "updated_at": time.time()}), encoding="utf-8")
            except OSError:
                logger.exception("Could not write phone session state")

    # ``from __future__ import annotations`` stores the nested WebSocket type
    # as a string. FastAPI cannot resolve a locally imported annotation, so
    # bind the concrete type before registering the route.
    audio_socket.__annotations__["websocket"] = WebSocket
    # Some reverse proxies preserve or add a trailing slash. Support both
    # forms so the phone client is not rejected during the WebSocket handshake.
    app.websocket("/ws/audio")(audio_socket)
    app.websocket("/ws/audio/")(audio_socket)

    return app


try:
    app = create_app()
except RuntimeError:
    # Keep audio-decoding utilities importable before optional web dependencies
    # are installed; starting the server will still report the missing package.
    app = None


if __name__ == "__main__":
    if app is None:
        raise SystemExit("Install FastAPI and Uvicorn with: python -m pip install -r requirements.txt")
    import uvicorn
    uvicorn.run("modules.phone_stream:app", host=os.getenv("KAIROS_HOST", "0.0.0.0"), port=int(os.getenv("KAIROS_PHONE_PORT", "8000")))
