from pathlib import Path
import json
from typing import List

WEIGHTS_FILE = Path(__file__).resolve().parent / 'weights.json'
DEFAULT_WEIGHTS: List[float] = [0.3, 0.4, 0.2, 0.2, 0.2, 0.2, -1.0, 0.1, -1.0, 0.1]


def load_weights() -> List[float]:
    if WEIGHTS_FILE.exists():
        try:
            return json.loads(WEIGHTS_FILE.read_text())
        except json.JSONDecodeError:
            pass
    return DEFAULT_WEIGHTS
