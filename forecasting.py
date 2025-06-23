"""Simple forecasting utilities using FFT and variance tracking."""

import json
import time
from pathlib import Path
from typing import Tuple

import numpy as np

from state import blend_ck, load_state, save_state
from statsmodels.tsa.ar_model import AutoReg
from arch import arch_model

DEFAULT_PARAMS = load_state().get("tuned_params", {})


def forecast_price(
    price_history: np.ndarray,
    lam: float | None = None,
    p_ar: int | None = None,
    q_res: int | None = None,
    garch: bool | None = None,
) -> Tuple[float, dict]:
    """Return a forecast using AR and optional GARCH."""
    mem = load_state()
    params = mem.get("tuned_params", {})
    if lam is None:
        lam = params.get("lam", 0.9)
    if p_ar is None:
        p_ar = params.get("p", 1)
    if q_res is None:
        q_res = params.get("q", 1)
    if garch is None:
        garch = params.get("garch", False)

    ar_mod = AutoReg(price_history, lags=p_ar, old_names=False).fit()
    ar_pred = float(
        ar_mod.predict(start=len(price_history), end=len(price_history))[-1]
    )
    resid = ar_mod.resid
    sigma2_new = float(np.var(resid))

    if garch:
        g_mod = arch_model(resid, p=1, q=1, rescale=False).fit(disp="off")
        sigma2_new = float(g_mod.forecast(horizon=1).variance.values[-1, 0])
        res_pred = float(g_mod.forecast(horizon=1).mean.values[-1, 0])
    elif q_res > 0:
        r_mod = AutoReg(resid, lags=q_res, old_names=False).fit()
        res_pred = float(r_mod.predict(start=len(resid), end=len(resid))[-1])
    else:
        res_pred = 0.0

    ck_new = np.fft.fft(price_history).astype(np.complex64)

    ck = blend_ck(np.array(mem.get("ck", ck_new), dtype=np.complex64), ck_new, lam)
    sigma2 = lam * sigma2_new + (1 - lam) * mem.get("sigma2", sigma2_new)

    forecast = ar_pred + res_pred
    mem.update({"ck": ck, "sigma2": sigma2})
    save_state(mem)
    return float(forecast), mem


def walk_forward_mae(
    series: np.ndarray,
    lam: float = 0.9,
    p_ar: int = 1,
    q_res: int = 1,
    garch: bool = False,
) -> float:
    preds = []
    for i in range(30, len(series)):
        window = series[i - 30 : i]
        ar_mod = AutoReg(window, lags=p_ar, old_names=False).fit()
        ar_pred = float(ar_mod.predict(start=len(window), end=len(window))[-1])
        resid = ar_mod.resid
        if garch:
            g_mod = arch_model(resid, p=1, q=1, rescale=False).fit(disp="off")
            res_pred = float(g_mod.forecast(horizon=1).mean.values[-1, 0])
        elif q_res > 0:
            r_mod = AutoReg(resid, lags=q_res, old_names=False).fit()
            res_pred = float(r_mod.predict(start=len(resid), end=len(resid))[-1])
        else:
            res_pred = 0.0
        preds.append(ar_pred + res_pred)
    if not preds:
        return float("inf")
    actual = series[30:]
    return float(np.mean(np.abs(np.array(preds) - actual)))
