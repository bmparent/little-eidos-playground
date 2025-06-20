from engine import QuantumEngine

class REPL:
    def __init__(self, engine: QuantumEngine, angle: float = 0.0):
        self.engine = engine
        self.env = {}
        self.angle = angle

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
            mapping = {'₿': 'X', '☀️': 'H', '📈': 'RZ'}
            gate = mapping.get(glyph)
            if gate == 'RZ':
                self.engine.apply('RZ', 0, theta=self.angle)
            elif gate:
                self.engine.apply(gate, 0)
            print('state:', self.engine.state)
            self.engine.show_probabilities()
        elif cmd == 'collapse':
            outcome = self.engine.measure()
            print('collapsed to', outcome)
            self.engine.show_probabilities()
        else:
            print(f'Unknown command: {cmd}')
