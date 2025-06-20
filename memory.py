"""Utilities for persisting a lightweight memory file."""

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

MEM_PATH = Path(".eidos_memory.json")


def load_memory() -> Optional[Dict]:
    """Return persisted memory if available."""
    if not MEM_PATH.exists():
        return None
    try:
        with MEM_PATH.open() as f:
            data = json.load(f)
        data["ck"] = np.array(data.get("ck", []), dtype=np.float32).view(np.complex64)
        data["state_amplitudes"] = np.array(
            data.get("state_amplitudes", []), dtype=np.float32
        ).view(np.complex64)
        return data
    except Exception:
        return None


def save_memory(mem: Dict) -> None:
    """Persist the ``mem`` dictionary to disk."""
    ck = mem.get("ck", np.array([], dtype=np.complex64))
    amps = mem.get("state_amplitudes", np.array([], dtype=np.complex64))
    data = {
        "timestamp": mem.get("timestamp"),
        "ck": ck.view(np.float32).tolist() if isinstance(ck, np.ndarray) else ck,
        "sigma2": float(mem.get("sigma2", 0.0)),
        "state_amplitudes": amps.view(np.float32).tolist()
        if isinstance(amps, np.ndarray)
        else amps,
    }
    with MEM_PATH.open("w") as f:
        json.dump(data, f)


def blend_ck(old: np.ndarray, new: np.ndarray, lam: float = 0.9) -> np.ndarray:
    """Blend arrays ``old`` and ``new`` with an EMA factor ``lam``."""
    if old.shape != new.shape:
        return new
    return lam * new + (1 - lam) * old
