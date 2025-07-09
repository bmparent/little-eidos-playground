"""Evaluate executed actions and log metrics."""

from pathlib import Path
import json
import time

METRIC_PATH = Path("metrics/actions.jsonl")
METRIC_PATH.parent.mkdir(exist_ok=True)


def log_metrics(data: dict) -> None:
    """Append ``data`` to the metrics log with a timestamp."""
    data["ts"] = int(time.time())
    with METRIC_PATH.open("a") as fp:
        fp.write(json.dumps(data) + "\n")


def evaluate(action: str, artefact: Path, surprise: float, success: bool) -> None:
    """Record the result of an action."""
    log_metrics(
        {
            "action": action,
            "artefact": str(artefact),
            "surprise": surprise,
            "success": success,
        }
    )
