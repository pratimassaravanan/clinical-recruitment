"""Simple site performance world-model helpers."""

from __future__ import annotations

from typing import Any, Dict


def predict_site_value(site: Dict[str, Any]) -> float:
    conversion = float(site.get("conversion_rate", 0.0))
    retention = float(site.get("retention_rate", 0.8))
    wait_days = float(site.get("avg_wait_days", 0.0))
    capacity = float(site.get("capacity_remaining", 0.0))
    return round(conversion * retention * (1.0 + capacity / 10.0) / max(1.0, wait_days), 4)
