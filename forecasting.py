"""Simple forecasting utilities using FFT and variance tracking."""

import time
from typing import Tuple

import numpy as np

from memory import blend_ck, load_memory, save_memory


def forecast_price(price_history: np.ndarray) -> Tuple[float, dict]:
    """Return a naive forecast and updated memory."""
    mem = load_memory()
    ck_new = np.fft.fft(price_history).astype(np.complex64)
    sigma2_new = float(np.var(price_history))

    if mem:
        ck = blend_ck(mem.get("ck", ck_new), ck_new)
        sigma2 = 0.9 * sigma2_new + 0.1 * mem.get("sigma2", sigma2_new)
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
