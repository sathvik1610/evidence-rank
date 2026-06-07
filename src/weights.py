"""
src/weights.py — Singleton weights loader

Loads weights.yaml once at import time and exposes a flat `W` dict.
All scorer/behavioral/feature modules import from here so that tuning
weights.yaml is the single source of truth without touching Python.

Usage:
    from src.weights import W
    threshold = W["social_proof.github_threshold"]
    mult = W["behavioral.inactive_heavy_mult"]
"""

import os
import yaml

_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "..", "weights.yaml")


def _flatten(d: dict, prefix: str = "") -> dict:
    """Recursively flatten a nested dict into dot-separated keys."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def _load() -> dict:
    path = os.path.abspath(_WEIGHTS_PATH)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"weights.yaml not found at {path}. "
            "This file must exist at the repository root."
        )
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _flatten(raw)


# Singleton — loaded once on first import
W: dict = _load()
