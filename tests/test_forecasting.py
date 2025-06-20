import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from forecasting import forecast_return


@pytest.mark.slow
def test_forecast_walkforward():
    try:
        import yfinance as yf
    except Exception:
        pytest.skip("yfinance missing")

    try:
        data = yf.download("BTC-USD", period="180d", interval="1d")
    except Exception:
        pytest.skip("no network")

    close = data["Close"].dropna()
    if len(close) < 100:
        pytest.skip("not enough data")

    errors_model = []
    errors_bh = []
    for i in range(100, len(close) - 1):
        train = close[:i]
        r_hat = forecast_return(train)
        pred = float(train.iloc[-1] * np.exp(r_hat))
        actual = float(close.iloc[i])
        errors_model.append(abs(pred - actual))
        errors_bh.append(abs(train.iloc[-1] - actual))

    assert np.mean(errors_model) <= np.mean(errors_bh)
