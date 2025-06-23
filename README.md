# little-eidos-playground

This repository contains a minimal Eidos parser and a small quantum engine.
The `emergent_intelligence.py` script fetches real time data, generates a
symbolic `.eidos` script, executes it, and commits the result.

![CI](https://github.com/bmparent/little-eidos-playground/actions/workflows/ci.yml/badge.svg)

- 🌐 **GCP Sense** – integrates the Global Consciousness Project RNG feed.

## 🤝 Alignment with ai-devops-buddy

This project tracks activity in the sibling `ai-devops-buddy` repository each
night. An alignment score is derived from recent commits, open issues and star
counts. When the score rises, a Visionary agent proposes roadmap updates.

## Usage

Install dependencies and run:

```bash
pip install -e .
python emergent_intelligence.py
```

The script will create `generated.eidos`, execute it line by line, display the
quantum state probabilities, and attempt to commit the generated file.
Persistent data such as tuned parameters and qubit amplitudes live in
`state.json`.

## Live Dashboard

A minimal Streamlit dashboard is provided under `ui/app.py`.
Launch it with:

```bash
pip install -e .[dev]
streamlit run ui/app.py
```

## 🔧 Optuna Tuner

Run the tuning agent to search forecasting hyper-parameters:

```bash
python -m agents.tuner
```

Results are stored in `state.json` under `tuned_params`.

## Quick-Deploy

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
pytest -q
streamlit run ui/app.py
```

![UI preview](docs/ui_preview.png)
## Glyph Mapping

Custom gate symbols are defined in `config/glyph_map.json`. The REPL loads this
file on each run allowing new glyphs to be added by pull request.
