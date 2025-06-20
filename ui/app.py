"""Streamlit dashboard for the Eidos playground."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import emergent_intelligence
from engine import QuantumToy
from memory import load_memory


st.set_page_config(page_title="Eidos Dashboard")

qubits = st.sidebar.slider("Qubits (1-4)", 1, 4, 1)

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
        lines = [l.strip() for l in f if l.strip()]
    st.text("\n".join(lines[-3:]))
except FileNotFoundError:
    st.write("No script yet.")
