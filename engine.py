import math
import cmath
import random
import numpy as np

class QuantumToy:
    def __init__(self):
        self.state = np.array([1+0j, 0+0j], dtype=complex)

    def entropy(self, level: float):
        phase = cmath.exp(1j * random.uniform(-level, level))
        self.state *= phase

    def apply_gate(self, gate_name: str):
        if gate_name == 'X':
            gate = np.array([[0, 1], [1, 0]], dtype=complex)
        elif gate_name == 'H':
            gate = (1 / math.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
        elif gate_name == 'I':
            gate = np.eye(2, dtype=complex)
        else:
            raise ValueError(f'Unknown gate {gate_name}')
        self.state = gate @ self.state

    def measure(self) -> int:
        probs = np.abs(self.state) ** 2
        outcome = random.choices([0, 1], weights=probs)[0]
        self.state = np.array([1, 0], dtype=complex) if outcome == 0 else np.array([0, 1], dtype=complex)
        return outcome

    def show_probabilities(self):
        import matplotlib.pyplot as plt
        probs = np.abs(self.state) ** 2
        plt.bar(['0', '1'], probs)
        plt.ylabel('Probability')
        plt.show()
