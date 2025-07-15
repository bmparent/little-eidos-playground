from __future__ import annotations
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import json

from .crawler import Artifact

CHEST = Path(__file__).resolve().parents[1] / "treasure-chest"


def stash(artifact: Artifact, score: float) -> Path:
    ts = datetime.utcnow().strftime("%Y/%m/%d/%H-%M-%S")
    dest = CHEST / ts
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "artifact.bin").write_bytes(artifact.content)
    meta = {"url": artifact.url, "sha256": artifact.sha256, "score": score}
    (dest / "meta.json").write_text(json.dumps(meta, indent=2))
    (dest / "why.md").write_text("\n".join([
        "# Reflection",
        "1. What surprised me?",
        "2. How might I play with this?",
        "3. Which rule(s) dominated the score?",
    ]))
    return dest
