"""Observe execution results and plan next steps."""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from typing import Dict

import yaml
import subprocess
import shadow

TODO_DIR = Path("todo")
EXPERIENCES_DIR = Path("experiences")
METRICS_FILE = Path("metrics/actions.jsonl")
OBSERVER_LOG = Path("metrics/observer.jsonl")


def _latest_card() -> Path | None:
    TODO_DIR.mkdir(exist_ok=True)
    cards = sorted(TODO_DIR.glob("*.yaml"))
    return cards[-1] if cards else None


def _latest_metrics() -> str:
    if METRICS_FILE.exists():
        with METRICS_FILE.open() as f:
            lines = f.readlines()
        return lines[-1].strip() if lines else ""
    return ""


def _latest_artefacts() -> list[Path]:
    if not EXPERIENCES_DIR.exists():
        return []
    dirs = sorted(EXPERIENCES_DIR.glob("*"))
    if not dirs:
        return []
    latest = dirs[-1]
    return list(latest.glob("*"))


def observe_and_plan(dry_run: bool = False) -> Dict:
    card_path = _latest_card()
    card_text = card_path.read_text() if card_path else ""
    metrics = _latest_metrics()
    artefacts = _latest_artefacts()
    artefact_names = [p.name for p in artefacts]

    prompt = (
        "Latest card:\n" + card_text + "\n" +
        "Metrics: " + metrics + "\n" +
        "Artefacts: " + ", ".join(artefact_names)
    )
    messages = [
        {"role": "system", "content": "You are the Observer agent."},
        {"role": "user", "content": prompt},
    ]
    plan = shadow.chat_with_shadow(messages)

    if not dry_run:
        if plan.get("type") == "patch" and plan.get("diff"):
            diff = plan["diff"]
            subprocess.run(["git", "apply", "--whitespace=fix", "-"],
                           input=diff.encode(), check=True)
        elif plan.get("type") == "card" and plan.get("card"):
            name = datetime.utcnow().strftime("%Y%m%d_%H%M%S_card.yaml")
            TODO_DIR.mkdir(exist_ok=True)
            (TODO_DIR / name).write_text(plan["card"])
    log = {
        "ts": datetime.utcnow().isoformat(),
        "card": card_path.name if card_path else None,
        "metric": metrics,
        "plan": plan,
    }
    OBSERVER_LOG.parent.mkdir(parents=True, exist_ok=True)
    with OBSERVER_LOG.open("a") as f:
        f.write(json.dumps(log) + "\n")
    return plan


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    observe_and_plan(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
