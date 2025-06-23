"""Simple forecasting utilities using FFT and variance tracking."""

import json
import time
from pathlib import Path
from typing import Tuple

import numpy as np

from memory import blend_ck, load_memory, save_memory

TUNE_FILE = Path('tuning.json')
if TUNE_FILE.exists():
    try:
        with TUNE_FILE.open() as f:
            _TUNE = json.load(f)
            DEFAULT_PARAMS = _TUNE.get('params', {})
    except Exception:
        DEFAULT_PARAMS = {}
else:
    DEFAULT_PARAMS = {}


def forecast_price(
    price_history: np.ndarray,
    lam: float | None = None,
    p_ar: int | None = None,
    q_res: int | None = None,
    garch: bool | None = None,
) -> Tuple[float, dict]:
    """Return a naive forecast and updated memory."""
    if lam is None:
        lam = DEFAULT_PARAMS.get('lam', 0.9)
    mem = load_memory()
    ck_new = np.fft.fft(price_history).astype(np.complex64)
    sigma2_new = float(np.var(price_history))

    if mem:
        ck = blend_ck(mem.get("ck", ck_new), ck_new, lam)
        sigma2 = lam * sigma2_new + (1 - lam) * mem.get("sigma2", sigma2_new)
    else:
        ck = ck_new
        sigma2 = sigma2_new

    forecast = float(price_history[-1] + ck[1].real if ck.size > 1 else price_history[-1])
    updated = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "ck": ck,
        "sigma2": sigma2,
        "state_amplitudes": mem.get("state_amplitudes") if mem else np.array([], dtype=np.complex64),
    }
    save_memory(updated)
    return forecast, updated


def walk_forward_mae(
    series: np.ndarray,
    lam: float = 0.9,
    p_ar: int = 1,
    q_res: int = 1,
    garch: bool = False,
) -> float:
    preds = []
    for i in range(30, len(series)):
        pred, _ = forecast_price(series[i - 30 : i], lam=lam, p_ar=p_ar, q_res=q_res, garch=garch)
        preds.append(pred)
    if not preds:
        return float('inf')
    actual = series[30:]
    return float(np.mean(np.abs(np.array(preds) - actual)))
