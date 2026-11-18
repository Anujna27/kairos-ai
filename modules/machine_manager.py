"""Machine registry and demonstration dashboard state."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List


MACHINES: List[Dict[str, str]] = [
    {"id": "MTR-01", "name": "Motor 01", "type": "Electric Motor"},
    {"id": "PMP-02", "name": "Pump 02", "type": "Industrial Pump"},
    {"id": "FAN-03", "name": "Fan 03", "type": "Cooling Fan"},
    {"id": "MTR-04", "name": "Motor 04", "type": "General Machine"},
]


def get_machines() -> List[Dict[str, str]]:
    """Return the configured machine registry."""
    return [dict(machine) for machine in MACHINES]


def machine_snapshot(history: list[dict]) -> List[Dict[str, object]]:
    """Merge session analysis history into machine cards."""
    snapshot = []
    for machine in MACHINES:
        records = [item for item in history if item.get("machine_id") == machine["id"]]
        latest = records[-1] if records else None
        snapshot.append({**machine, "status": latest.get("result", "NOT ANALYZED") if latest else "NOT ANALYZED", "last_analysis": latest.get("timestamp", "—") if latest else "—", "anomaly_score": latest.get("score") if latest else None})
    return snapshot


def utc_timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
