"""Streamlit dashboard for the Eidos playground."""

import argparse
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import emergent_intelligence
from engine import QuantumToy
from state import load_state


parser = argparse.ArgumentParser()
parser.add_argument("--watch", action="store_true")
opts, _ = parser.parse_known_args()

st.set_page_config(page_title="Eidos Dashboard")

if opts.watch:

    def _refresh():
        time.sleep(30)
        st.experimental_rerun()

    threading.Thread(target=_refresh, daemon=True).start()

qubits = st.sidebar.slider("Qubits (1-4)", 1, 4, 1)

radar_path = Path("visual/radar_latest.png")
if radar_path.exists():
    st.sidebar.image(str(radar_path))

try:
    with open(Path("config/runes.json")) as f:
        meanings = json.load(f)
except FileNotFoundError:
    meanings = {}

state = load_state()
last = state.get("last_rune")
if last:
    meaning = meanings.get(last, {}).get("meaning", "")
    st.sidebar.write(f"Last rune: {last} – {meaning}")

os.environ["EIDOS_AUTOPUSH"] = "0"
emergent_intelligence.main()

engine = QuantumToy(n=qubits)
fig = engine.plot_bloch() if qubits == 1 else engine.plot_heatmap()
st.pyplot(fig)

amps = engine.state
basis_labels = [format(i, f"0{qubits}b") for i in range(len(amps))]
probs = np.abs(amps) ** 2

df = pd.DataFrame({"basis": basis_labels, "probability": probs})
df = df.sort_values("probability", ascending=False).head(8)
st.table(df)

try:
    with open("generated.eidos", encoding="utf8") as f:
        lines = [line.strip() for line in f if line.strip()]
    st.text("\n".join(lines[-3:]))
except FileNotFoundError:
    st.write("No script yet.")
