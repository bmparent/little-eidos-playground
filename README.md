# little-eidos-playground

This repository contains a minimal Eidos parser and a simple one-qubit toy engine
for experimentation. The `emergent_intelligence.py` script fetches real time data,
generates a symbolic `.eidos` script, executes it, and commits the result.

## Usage

Install dependencies and run:

```bash
pip install -r requirements.txt
python emergent_intelligence.py
```

The script will create `generated.eidos`, execute it line by line, display the
quantum state probabilities, and attempt to commit the generated file.

## Live Dashboard

A minimal Streamlit dashboard is provided under `ui/app.py`.
Launch it with:

```bash
pip install -r requirements.txt
streamlit run ui/app.py
```

![UI preview](docs/ui_preview.png)

## \ud83d\udd10 GitHub Token Setup

Several helper agents interact with the GitHub API. Create a Personal Access
Token with scopes `repo:status`, `delete_repo`, and `public_repo` and add it to
this repository as the secret `GH_BOT_TOKEN`. Locally, export the same token in
`GH_BOT_TOKEN` so utilities such as `agents.branch_pruner` can authenticate.

The agents now rely on `gh_api.py` rather than invoking `git` directly for
branch enumeration and deletion.
