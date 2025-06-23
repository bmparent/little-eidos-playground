import json
import numpy as np
from math import pi
from pathlib import Path
from engine import QuantumToy
from sensors import gcp_rng

try:
    with open(Path(__file__).parent / "config" / "glyph_map.json") as fh:
        _map = json.load(fh)
except FileNotFoundError:
    _map = {"₿": "X", "☀️": "H", "📈": "I", "⚡": "RZ", "🤝": "RY"}

GLYPH_TO_GATE = _map


def glyph_map():
    return GLYPH_TO_GATE


class REPL:
    def __init__(self, engine: QuantumToy):
        self.engine = engine
        self.env = {}
        self.mapping = GLYPH_TO_GATE

    def execute(self, stmt):
        if stmt is None:
            return
        cmd = stmt[0]
        if cmd == "let":
            _, name, value = stmt
            self.env[name] = value
        elif cmd == "entropy":
            _, level = stmt
            self.engine.entropy(level)
        elif cmd == "observe":
            _, name = stmt
            glyph = self.env.get(name)
            if glyph is None:
                print(f"Unknown symbol {name}")
                return
            spec = self.mapping.get(glyph)
            if not spec:
                print(f"Unknown glyph {glyph}")
                return
            if isinstance(spec, dict):
                gate = spec.get("gate")
                param = spec.get("param")
            else:
                gate = spec
                if gate == "RZ":
                    param = "rng_z"
                elif gate == "RY":
                    param = "alignment"
                else:
                    param = None
            theta = None
            if param == "rng_z":
                z = gcp_rng.latest_z()
                theta = np.clip(z, -5, 5) * pi / 10
            elif param == "alignment":
                score = self.env.get("alignment_score", 0.0)
                theta = np.clip(score, -5, 5) * pi / 50
            if gate in {"RY"}:
                self.engine.apply(gate, 1, theta=theta)
            else:
                self.engine.apply(gate, 0, theta=theta)
            print("state:", self.engine.state)
            self.engine.show_probabilities()
        elif cmd == "glyph":
            symbol = stmt[1]
            spec = self.mapping.get(symbol)
            if not spec:
                print(f"Unknown glyph {symbol}")
                return
            if isinstance(spec, dict):
                gate = spec.get("gate")
                param = spec.get("param")
            else:
                gate = spec
                if gate == "RZ":
                    param = "rng_z"
                elif gate == "RY":
                    param = "alignment"
                else:
                    param = None
            theta = None
            if param == "rng_z":
                z = gcp_rng.latest_z()
                theta = np.clip(z, -5, 5) * pi / 10
            elif param == "alignment":
                score = self.env.get("alignment_score", 0.0)
                theta = np.clip(score, -5, 5) * pi / 50
            if gate in {"RY"}:
                self.engine.apply(gate, 1, theta=theta)
            else:
                self.engine.apply(gate, 0, theta=theta)
            print("state:", self.engine.state)
            self.engine.show_probabilities()
        elif cmd == "collapse":
            outcome = self.engine.measure()
            print("collapsed to", outcome)
            self.engine.show_probabilities()
        else:
            print(f"Unknown command: {cmd}")
