import numpy as np
from math import pi
from engine import QuantumToy
from sensors import gcp_rng

class REPL:
    def __init__(self, engine: QuantumToy):
        self.engine = engine
        self.env = {}

    def execute(self, stmt):
        if stmt is None:
            return
        cmd = stmt[0]
        if cmd == 'let':
            _, name, value = stmt
            self.env[name] = value
        elif cmd == 'entropy':
            _, level = stmt
            self.engine.entropy(level)
        elif cmd == 'observe':
            _, name = stmt
            glyph = self.env.get(name)
            if glyph is None:
                print(f'Unknown symbol {name}')
                return
            if glyph == '⚡':
                z = gcp_rng.latest_z()
                theta = np.clip(z, -5, 5) * pi / 10
                self.engine.apply_gate('RZ', theta=theta)
            elif glyph == '🤝':
                score = self.env.get('alignment_score', 0.0)
                theta = np.clip(score, -5, 5) * pi / 50
                self.engine.apply_gate('RY', qubit=1, theta=theta)
            else:
                mapping = {'₿': 'X', '☀️': 'H', '📈': 'I'}
                gate = mapping.get(glyph)
                if gate:
                    self.engine.apply_gate(gate)
            print('state:', self.engine.state)
            self.engine.show_probabilities()
        elif cmd == 'glyph':
            symbol = stmt[1]
            if symbol == '⚡':
                z = gcp_rng.latest_z()
                theta = np.clip(z, -5, 5) * pi / 10
                self.engine.apply_gate('RZ', theta=theta)
            elif symbol == '🤝':
                score = self.env.get('alignment_score', 0.0)
                theta = np.clip(score, -5, 5) * pi / 50
                self.engine.apply_gate('RY', qubit=1, theta=theta)
            else:
                gate = {'H': 'H', 'X': 'X', 'I': 'I'}.get(symbol)
                if gate:
                    self.engine.apply_gate(gate)
            print('state:', self.engine.state)
            self.engine.show_probabilities()
        elif cmd == 'collapse':
            outcome = self.engine.measure()
            print('collapsed to', outcome)
            self.engine.show_probabilities()
        else:
            print(f'Unknown command: {cmd}')
