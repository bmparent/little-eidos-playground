import json
import time
import subprocess
import os
import requests

from engine import QuantumToy
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


def run_script():
    q = QuantumToy()
    repl = REPL(q)
    with open("generated.eidos", encoding="utf8") as f:
        for line in f:
            if line.strip():
                print(f"[run] >> {line.strip()}")
                stmt = Parser(line).parse()
                repl.execute(stmt)


def auto_commit():
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    subprocess.run(["git", "add", "generated.eidos"], check=False)
    subprocess.run(["git", "commit", "-m", f"chore: update generated.eidos at {timestamp}"], check=False)
    if os.getenv("EIDOS_AUTOPUSH", "1") != "0":
        subprocess.run(["git", "push", "origin", "feature/emergent-playground"], check=False)


def main():
    freq_glyph, freq_const = fetch_bitcoin()
    vib_glyph, vib_const = fetch_weather()
    energy_glyph, energy_const = fetch_trending()

    generate_script(freq_glyph, freq_const, vib_glyph, vib_const, energy_const)
    run_script()
    auto_commit()


if __name__ == "__main__":
    main()
