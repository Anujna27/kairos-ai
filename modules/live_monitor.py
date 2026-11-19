"""Continuous microphone monitoring using the trained KAIROS model."""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from .alerts import AlertPolicy, EmailConfig, EmailNotifier, EventStore
from .anomaly_detection import DEFAULT_MODEL_PATH, load_model, predict_from_features
from .feature_extraction import SAMPLE_RATE, extract_features_from_audio

logger = logging.getLogger(__name__)


class LiveMonitor:
    """Capture and analyze overlapping audio windows until stopped."""

    def __init__(self, *, machine_id: str, model_path: str | Path = DEFAULT_MODEL_PATH,
                 event_db: str | Path = "data/kairos_events.db", window_seconds: float = 5.0,
                 interval_seconds: float = 5.0, alert_policy: AlertPolicy | None = None,
                 notifier: EmailNotifier | None = None,
                 waveform_path: str | Path = "data/live_waveform.json") -> None:
        if window_seconds <= 0 or interval_seconds <= 0:
            raise ValueError("Window and interval durations must be positive")
        self.machine_id = machine_id
        self.window_samples = int(window_seconds * SAMPLE_RATE)
        self.interval_seconds = interval_seconds
        self.model = load_model(model_path)
        self.store = EventStore(event_db)
        self.policy = alert_policy or AlertPolicy()
        self.notifier = notifier
        self.waveform_path = Path(waveform_path)
        self.waveform_path.parent.mkdir(parents=True, exist_ok=True)
        self.buffer: deque[np.ndarray] = deque(maxlen=self.window_samples)
        self.last_analysis = time.monotonic()

    def close(self) -> None:
        self.store.close()

    def analyze(self, audio: Any) -> dict[str, Any]:
        samples = np.asarray(audio, dtype=np.float32)
        stride = max(1, len(samples) // 500)
        waveform = samples[::stride].tolist()
        temporary = self.waveform_path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"timestamp": time.time(), "samples": waveform}), encoding="utf-8")
        temporary.replace(self.waveform_path)
        features = extract_features_from_audio(samples, SAMPLE_RATE)
        result = predict_from_features(features, self.model)
        alert_sent = self.policy.observe(bool(result["is_anomaly"]), time.monotonic())
        message = "Potential acoustic anomaly" if result["is_anomaly"] else "Normal"
        if alert_sent and self.notifier is not None:
            try:
                self.notifier.send_anomaly(
                    machine_id=self.machine_id,
                    anomaly_score=float(result["anomaly_score"]),
                    decision_score=float(result["decision_score"]),
                    consecutive_anomalies=self.policy.consecutive_anomalies,
                )
            except Exception:
                logger.exception("Could not send anomaly email")
                alert_sent = False
                message = "Potential acoustic anomaly; email delivery failed"
        self.store.record(
            machine_id=self.machine_id,
            status=str(result["status"]),
            anomaly_score=float(result["anomaly_score"]),
            decision_score=float(result["decision_score"]),
            alert_sent=alert_sent,
            message=message,
        )
        logger.info("%s: %s score=%.6f", self.machine_id, result["status"], result["anomaly_score"])
        return {**result, "alert_sent": alert_sent}

    def run(self, *, device: int | str | None = None) -> None:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError("Live monitoring requires sounddevice.") from exc

        def callback(indata: np.ndarray, frames: int, callback_time: Any, status: Any) -> None:
            if status:
                logger.warning("Audio input status: %s", status)
            self.buffer.extend(indata[:, 0].astype(np.float32).tolist())

        logger.info("Starting live monitor for %s", self.machine_id)
        with sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            blocksize=max(256, int(SAMPLE_RATE * min(self.interval_seconds, 1.0))),
            device=device, callback=callback,
        ):
            while True:
                time.sleep(0.25)
                if len(self.buffer) < self.window_samples:
                    continue
                now = time.monotonic()
                if now - self.last_analysis < self.interval_seconds:
                    continue
                self.last_analysis = now
                audio = np.asarray(self.buffer, dtype=np.float32)
                try:
                    self.analyze(audio)
                except Exception:
                    logger.exception("Live audio analysis failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine-id", default="MTR-01")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--event-db", type=Path, default=Path("data/kairos_events.db"))
    parser.add_argument("--waveform", type=Path, default=Path("data/live_waveform.json"))
    parser.add_argument("--window-seconds", type=float, default=5.0)
    parser.add_argument("--interval-seconds", type=float, default=5.0)
    parser.add_argument("--required-anomalies", type=int, default=3)
    parser.add_argument("--cooldown-seconds", type=int, default=3600)
    parser.add_argument("--no-email", action="store_true", help="Log anomalies without sending email")
    parser.add_argument("--device", default=None, help="Optional sounddevice input device index/name")
    args = parser.parse_args()

    notifier = None if args.no_email else EmailNotifier(EmailConfig.from_environment())
    monitor = LiveMonitor(
        machine_id=args.machine_id, model_path=args.model, event_db=args.event_db,
        window_seconds=args.window_seconds, interval_seconds=args.interval_seconds,
        alert_policy=AlertPolicy(args.required_anomalies, args.cooldown_seconds),
        notifier=notifier, waveform_path=args.waveform,
    )
    try:
        monitor.run(device=args.device)
    except KeyboardInterrupt:
        logger.info("Live monitor stopped")
    finally:
        monitor.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
