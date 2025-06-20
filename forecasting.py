"""Energy-Frequency Forecasting 2.0 implementation."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch import arch_model
from numpy.fft import fft, ifft


def forecast_return(
    series: pd.Series,
    lookback: int = 90,
    lam: float = 0.9,
    p_ar: int = 2,
    q_res: int = 1,
    garch: bool = True,
) -> float:
    """Forecast next-step log return using the energy-frequency algorithm."""

    price = series.dropna()
    r = np.log(price).diff().dropna()
    if len(r) <= lookback + max(p_ar, q_res):
        raise ValueError("series too short for chosen lookback")

    # Layer 1: pre-whiten with AR(p)
    ar_mod = sm.tsa.AutoReg(r, lags=p_ar, old_names=False).fit()
    resid = ar_mod.resid[-lookback:]

    L = len(resid)
    omega = 2 * np.pi * np.arange(L) / L

    # Layer 2: adaptive spectrum update
    c = fft(resid)
    c_next = lam * c * np.exp(1j * omega)
    fft_pred = ifft(c_next).real[0]

    # Residuals
    e = resid - ifft(c).real

    # Layer 3: AR polish on residuals
    ar_res_mod = sm.tsa.AutoReg(e, lags=q_res, old_names=False).fit()
    ar_res = ar_res_mod.predict(start=len(e), end=len(e))[0]

    r_hat = float(fft_pred + ar_res)

    # Layer 4: GARCH variance forecast (not used in mean)
    if garch:
        try:
            g_mod = arch_model(e * 100, p=1, q=1)
            g_res = g_mod.fit(disp="off")
            _ = g_res.forecast(horizon=1)
        except Exception:
            pass

    return r_hat


def forecast_price(series: pd.Series, **kwargs: Optional[float]) -> float:
    """Forecast next price level using ``forecast_return``."""

    r_hat = forecast_return(series, **kwargs)
    return float(series.iloc[-1] * np.exp(r_hat))
