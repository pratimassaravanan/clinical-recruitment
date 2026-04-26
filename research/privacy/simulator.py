"""Federated/privacy-preserving patient simulation scaffolds."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


def anonymize_patient_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    anonymized: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        cleaned = dict(row)
        if "id" in cleaned:
            cleaned["id"] = f"anon-{idx:04d}"
        anonymized.append(cleaned)
    return anonymized
