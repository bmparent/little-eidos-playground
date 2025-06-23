# little-eidos-playground

This repository contains a minimal Eidos parser and a simple one-qubit toy engine
for experimentation. The `emergent_intelligence.py` script fetches real time data,
generates a symbolic `.eidos` script, executes it, and commits the result.

## Usage

Install dependencies and run:

```bash
pip install -e .
python emergent_intelligence.py
```

The script will create `generated.eidos`, execute it line by line, display the
quantum state probabilities, and attempt to commit the generated file.

## Live Dashboard

A minimal Streamlit dashboard is provided under `ui/app.py`.
Launch it with:

```bash
pip install -e .[dev]
streamlit run ui/app.py
```

## Quick-Deploy

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
pytest -q
streamlit run ui/app.py
```

![UI preview](docs/ui_preview.png)
