import json
import os
import time
from datetime import datetime
import subprocess
import requests

import pandas as pd
import numpy as np
from engine import QuantumEngine
from forecasting import forecast_price, forecast_return
from parser import Parser
from repl import REPL


def fetch_bitcoin():
    try:
        resp = requests.get(
            "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd",
            timeout=10,
        )
        data = resp.json()
        price = data["bitcoin"]["usd"]
        freq_const = price / 10000
        print(f"Fetched Bitcoin: {price} -> freq_const={freq_const}")
        return "₿", freq_const
    except Exception as e:
        print(f"Error fetching Bitcoin price: {e}")
        price = 50000
        freq_const = price / 10000
        print(f"Using default price {price} -> freq_const={freq_const}")
        return "₿", freq_const


def fetch_btc_series(days: int = 120) -> pd.Series:
    """Retrieve historical BTC prices for forecasting."""
    try:
        import yfinance as yf

        data = yf.download("BTC-USD", period=f"{days}d", interval="1d")
        close = data["Close"].dropna()
        print(f"Fetched {len(close)} BTC prices for forecasting")
        return close
    except Exception as e:
        print(f"BTC history fetch failed: {e}")
        return pd.Series(dtype=float)


def fetch_weather(lat="0", lon="0"):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        temp_c = data["current_weather"]["temperature"]
        code = data["current_weather"].get("weathercode", 0)
        clear_codes = {0, 1, 2, 3}
        glyph = "☀️" if code in clear_codes else "🌧️"
        vib_const = temp_c / 10
        print(
            f"Fetched weather: {temp_c}°C code {code} -> {glyph} vib_const={vib_const}"
        )
        return glyph, vib_const
    except Exception as e:
        print(f"Error fetching weather: {e}")
        temp_c = 20
        vib_const = temp_c / 10
        print(f"Using default temp {temp_c} -> vib_const={vib_const}")
        return "☀️", vib_const


def fetch_trending():
    try:
        resp = requests.get(
            "https://api.reddit.com/r/all/top?limit=1&t=day",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        data = resp.json()
        title = data["data"]["children"][0]["data"]["title"]
        energy_const = len(title) % 10
        print(f"Fetched trending title: {title!r} -> energy_const={energy_const}")
        return "📈", energy_const
    except Exception as e:
        print(f"Error fetching trending topic: {e}")
        title = "Hello"
        energy_const = len(title) % 10
        print(f"Using default title {title!r} -> energy_const={energy_const}")
        return "📈", energy_const


def generate_script(freq_glyph, freq_const, vib_glyph, vib_const, energy_const):
    with open("generated.eidos", "w", encoding="utf8") as f:
        f.write("# Symbolic script created at runtime\n")
        f.write(f"let Frequency = {freq_glyph}\n")
        f.write(f"entropy {freq_const}\n")
        f.write("observe Frequency\n\n")
        f.write(f"let Vibration = {vib_glyph}\n")
        f.write(f"entropy {vib_const}\n")
        f.write("observe Vibration\n\n")
        f.write("let Energy = 📈\n")
        f.write(f"entropy {energy_const}\n")
        f.write("observe Energy\n\n")
        f.write("collapse\n")


def run_script(theta: float) -> None:
    q = QuantumEngine(1)
    repl = REPL(q, angle=theta)
    with open("generated.eidos", encoding="utf8") as f:
        for line in f:
            if line.strip():
                print(f"[run] >> {line.strip()}")
                stmt = Parser(line).parse()
                repl.execute(stmt)


def auto_commit() -> None:
    """Commit and push ``generated.eidos`` to an auto branch."""

    if os.environ.get("EIDOS_AUTOPUSH", "1") == "0":
        print("Autopush disabled via EIDOS_AUTOPUSH")
        return

    if subprocess.run(["git", "diff", "--quiet"]).returncode != 0:
        print("Repository not clean; skipping push")
        return

    ts = datetime.utcnow().replace(microsecond=0)
    branch = f"autogen/{ts:%Y-%m-%d-%H%M}"
    msg = f"\U0001F916 auto-growth: synth\u2010eidos at {ts.isoformat()}"

    subprocess.run(["git", "checkout", "-b", branch], check=True)
    subprocess.run(["git", "add", "generated.eidos"], check=True)
    subprocess.run(["git", "commit", "-m", msg], check=True)
    result = subprocess.run(["git", "push", "-u", "origin", branch])
    if result.returncode != 0:
        raise RuntimeError("Push rejected")


def main():
    freq_glyph, freq_const = fetch_bitcoin()
    vib_glyph, vib_const = fetch_weather()
    energy_glyph, energy_const = fetch_trending()

    series = fetch_btc_series()
    theta = 0.0
    if len(series) > 0:
        try:
            r_hat = forecast_return(series)
            theta = float(np.clip(r_hat * np.pi, -np.pi, np.pi))
        except Exception as e:
            print(f"Forecast failed: {e}")

    generate_script(freq_glyph, freq_const, vib_glyph, vib_const, energy_const)
    run_script(theta)
    auto_commit()


if __name__ == "__main__":
    main()
