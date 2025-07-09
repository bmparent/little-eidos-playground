"""Execute pending Action Cards."""

import yaml
from importlib import import_module
from pathlib import Path
import os
from dataclasses import dataclass

from datetime import datetime
from evaluator import evaluate
from curiosity import calc_surprise
from safety import within_budget, global_pause


@dataclass
class ActionCard:
    kind: str
    name: str
    rationale: str = ""
    success: str = ""

    @classmethod
    def from_path(cls, path: Path) -> "ActionCard":
        with path.open() as f:
            data = yaml.safe_load(f)
        return cls(
            kind=data.get("kind", ""),
            name=data.get("name", ""),
            rationale=data.get("rationale", ""),
            success=data.get("success", ""),
        )

TODO_DIR = Path("todo")
EXPERIENCES_DIR = Path("experiences")


def next_card() -> Path | None:
    TODO_DIR.mkdir(exist_ok=True)
    cards = sorted(TODO_DIR.glob("*.yaml"))
    return cards[0] if cards else None


def execute(card_path: Path) -> None:
    card = ActionCard.from_path(card_path)

    module_name = f"{card.kind}s.{card.name}"
    result = None

    try:
        mod = import_module(module_name)
        if hasattr(mod, "main"):
            result = mod.main()
    except Exception as e:
        result = {"error": str(e)}

    exp_dir = EXPERIENCES_DIR / datetime.utcnow().strftime("%Y-%m-%d")
    exp_dir.mkdir(parents=True, exist_ok=True)
    artefact = exp_dir / f"{card_path.stem}.txt"
    with artefact.open("w") as f:
        f.write(str(result))

    surprise = calc_surprise(
        os.urandom(4)
    )  # dummy vector for now
    success = result is not None
    evaluate(card_path.name, artefact, surprise, success)
    card_path.unlink()


def main() -> None:
    if global_pause():
        print("Execution paused via issue flag")
        return
    card = next_card()
    if not card:
        print("No pending cards")
        return
    if not within_budget("openai", 0):
        print("Budget exceeded")
        return
    execute(card)


if __name__ == "__main__":
    main()
