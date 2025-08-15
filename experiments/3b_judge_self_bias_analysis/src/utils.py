"""Utility helpers for Experiment 3b (placeholder for future use)."""

from __future__ import annotations

from pathlib import Path


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)
