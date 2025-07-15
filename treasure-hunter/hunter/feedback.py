from pathlib import Path
import json
from typing import List

WEIGHTS_FILE = Path(__file__).resolve().parent / 'weights.json'


def update_weights(delta: List[float]) -> None:
    if WEIGHTS_FILE.exists():
        try:
            weights = json.loads(WEIGHTS_FILE.read_text())
        except json.JSONDecodeError:
            weights = []
    else:
        weights = []
    for i, d in enumerate(delta):
        if i < len(weights):
            weights[i] += d
        else:
            weights.append(d)
    WEIGHTS_FILE.write_text(json.dumps(weights))
