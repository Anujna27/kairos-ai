"""Alert policies, SQLite event storage, and SMTP notifications."""

from __future__ import annotations

import os
import sqlite3
import smtplib
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class EmailConfig:
    host: str
    port: int
    username: str
    password: str
    sender: str
    recipient: str
    use_tls: bool = True

    @classmethod
    def from_environment(cls) -> "EmailConfig":
        required = {
            "SMTP_HOST": os.getenv("SMTP_HOST", ""),
            "SMTP_USERNAME": os.getenv("SMTP_USERNAME", ""),
            "SMTP_PASSWORD": os.getenv("SMTP_PASSWORD", ""),
            "ALERT_FROM": os.getenv("ALERT_FROM", ""),
            "ALERT_TO": os.getenv("ALERT_TO", ""),
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"Missing email environment variables: {', '.join(missing)}")
        return cls(
            host=required["SMTP_HOST"],
            port=int(os.getenv("SMTP_PORT", "587")),
            username=required["SMTP_USERNAME"],
            password=required["SMTP_PASSWORD"],
            sender=required["ALERT_FROM"],
            recipient=required["ALERT_TO"],
            use_tls=os.getenv("SMTP_USE_TLS", "true").lower() not in {"0", "false", "no"},
        )


class EventStore:
    """Small durable event store suitable for a single monitor process."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Live phone processing runs feature extraction in a worker thread.
        # Allow this store connection to be used by that worker.
        self.connection = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                machine_id TEXT NOT NULL,
                status TEXT NOT NULL,
                anomaly_score REAL,
                decision_score REAL,
                alert_sent INTEGER NOT NULL DEFAULT 0,
                message TEXT
            )"""
        )
        self.connection.commit()

    def record(self, *, machine_id: str, status: str, anomaly_score: float | None,
               decision_score: float | None, alert_sent: bool, message: str) -> None:
        self.connection.execute(
            """INSERT INTO events
               (timestamp, machine_id, status, anomaly_score, decision_score, alert_sent, message)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (utc_now(), machine_id, status, anomaly_score, decision_score, int(alert_sent), message),
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """SELECT timestamp, machine_id, status, anomaly_score,
                      decision_score, alert_sent, message
               FROM events ORDER BY id DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        columns = ["timestamp", "machine_id", "status", "anomaly_score",
                   "decision_score", "alert_sent", "message"]
        return [dict(zip(columns, row)) for row in rows]

    def clear(self) -> None:
        self.connection.execute("DELETE FROM events")
        self.connection.commit()


class EmailNotifier:
    def __init__(self, config: EmailConfig) -> None:
        self.config = config

    def send_anomaly(self, *, machine_id: str, anomaly_score: float, decision_score: float,
                     consecutive_anomalies: int) -> None:
        message = EmailMessage()
        message["Subject"] = f"KAIROS anomaly alert: {machine_id}"
        message["From"] = self.config.sender
        message["To"] = self.config.recipient
        message.set_content(
            f"KAIROS detected a potential acoustic anomaly.\n\n"
            f"Machine: {machine_id}\n"
            f"Time (UTC): {utc_now()}\n"
            f"Anomaly score: {anomaly_score:.6f}\n"
            f"Decision score: {decision_score:.6f}\n"
            f"Consecutive anomalous windows: {consecutive_anomalies}\n\n"
            "This is an anomaly signal, not a failure diagnosis."
        )
        with smtplib.SMTP(self.config.host, self.config.port, timeout=30) as smtp:
            if self.config.use_tls:
                smtp.starttls()
            smtp.login(self.config.username, self.config.password)
            smtp.send_message(message)


class AlertPolicy:
    """Require repeated anomalies and suppress duplicate email alerts."""

    def __init__(self, required_anomalies: int = 3, cooldown_seconds: int = 3600) -> None:
        if required_anomalies < 1 or cooldown_seconds < 0:
            raise ValueError("Alert policy values must be positive")
        self.required_anomalies = required_anomalies
        self.cooldown_seconds = cooldown_seconds
        self.consecutive_anomalies = 0
        self.last_alert_at: float | None = None

    def observe(self, is_anomaly: bool, now: float) -> bool:
        if is_anomaly:
            self.consecutive_anomalies += 1
        else:
            self.consecutive_anomalies = 0
        ready = self.consecutive_anomalies >= self.required_anomalies
        cooled_down = self.last_alert_at is None or now - self.last_alert_at >= self.cooldown_seconds
        if ready and cooled_down:
            self.last_alert_at = now
            return True
        return False
