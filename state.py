from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

import numpy as np
import portalocker

STATE_PATH = Path("state.json")
OLD_MEM = Path(".eidos_memory.json")
TUNE_FILE = Path("tuning.json")

DEFAULT_STATE: Dict = {
    "version": 3,
    "qubit_state": [],
    "ck": [],
    "sigma2": 0.0,
    "tuned_params": {"lam": 0.9, "p": 1, "q": 1, "garch": False},
    "mae": 0.0,
    "last_alignment": 0.0,
}


def _migrate() -> Dict:
    state = DEFAULT_STATE.copy()
    if OLD_MEM.exists():
        try:
            with OLD_MEM.open() as f:
                data = json.load(f)
            state["ck"] = data.get("ck", state["ck"])
            state["sigma2"] = data.get("sigma2", state["sigma2"])
            state["qubit_state"] = data.get("state_amplitudes", state["qubit_state"])
            state["mae"] = data.get("mae", state["mae"])
            OLD_MEM.unlink()
        except Exception:
            pass
    if TUNE_FILE.exists():
        try:
            with TUNE_FILE.open() as f:
                t = json.load(f)
            if isinstance(t, dict) and "params" in t:
                state["tuned_params"] = t["params"]
            TUNE_FILE.unlink()
        except Exception:
            pass
    return state


def load_state() -> Dict:
    if not STATE_PATH.exists():
        state = _migrate()
        save_state(state)
        return state
    with STATE_PATH.open("r") as f:
        portalocker.lock(f, portalocker.LOCK_SH)
        data = json.load(f)
        portalocker.unlock(f)
    data.setdefault("version", 3)
    data["ck"] = np.array(data.get("ck", []), dtype=np.float32).view(np.complex64)
    data["qubit_state"] = np.array(data.get("qubit_state", []), dtype=np.float32).view(
        np.complex64
    )
    return data


def save_state(state: Dict) -> None:
    ck = state.get("ck", np.array([], dtype=np.complex64))
    qstate = state.get("qubit_state", np.array([], dtype=np.complex64))
    data = dict(state)
    data["ck"] = ck.view(np.float32).tolist() if isinstance(ck, np.ndarray) else ck
    data["qubit_state"] = (
        qstate.view(np.float32).tolist() if isinstance(qstate, np.ndarray) else qstate
    )
    with STATE_PATH.open("w") as f:
        portalocker.lock(f, portalocker.LOCK_EX)
        json.dump(data, f)
        f.flush()
        os.fsync(f.fileno())
        portalocker.unlock(f)


def blend_ck(old: np.ndarray, new: np.ndarray, lam: float = 0.9) -> np.ndarray:
    if old.shape != new.shape:
        return new
    return lam * new + (1 - lam) * old
