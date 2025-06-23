"""Simple quantum engine supporting up to four qubits."""

import math
import random
from typing import List

import numpy as np

from memory import load_memory, save_memory


def _single_qubit_gate(gate: np.ndarray, n: int, target: int) -> np.ndarray:
    """Construct a full gate matrix acting on ``target`` qubit among ``n``."""
    result = np.array([[1]], dtype=np.complex64)
    for i in range(n):
        if i == target:
            result = np.kron(result, gate)
        else:
            result = np.kron(result, np.eye(2, dtype=np.complex64))
    return result


class QuantumToy:
    """Lightweight quantum simulator."""

    def __init__(self, n: int = 1):
        self.n = n
        dim = 2 ** n
        self.state = np.zeros(dim, dtype=np.complex64)
        self.state[0] = 1 + 0j
        mem = load_memory()
        if mem is not None:
            amps = mem.get("state_amplitudes")
            if isinstance(amps, np.ndarray) and amps.shape == (dim,):
                self.state = amps

    # ------------------------------------------------------------------ gates
    def entropy(self, level: float) -> None:
        phase = np.exp(1j * random.uniform(-level, level))
        self.state *= phase

    def apply_gate(self, gate_name: str, qubit: int = 0, theta: float | None = None) -> None:
        gates = {
            "X": np.array([[0, 1], [1, 0]], dtype=np.complex64),
            "H": (1 / math.sqrt(2))
            * np.array([[1, 1], [1, -1]], dtype=np.complex64),
            "I": np.eye(2, dtype=np.complex64),
        }
        if gate_name == "RZ":
            if theta is None:
                raise ValueError("RZ gate requires theta")
            gates["RZ"] = np.array(
                [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]],
                dtype=np.complex64,
            )
        if gate_name == "RY":
            if theta is None:
                raise ValueError("RY gate requires theta")
            gates["RY"] = np.array(
                [[np.cos(theta / 2), -np.sin(theta / 2)],
                 [np.sin(theta / 2), np.cos(theta / 2)]],
                dtype=np.complex64,
            )
        gate = gates.get(gate_name)
        if gate is None:
            raise ValueError(f"Unknown gate {gate_name}")
        full_gate = _single_qubit_gate(gate, self.n, qubit)
        self.state = full_gate @ self.state

    def show_probabilities(self) -> None:
        probs = np.abs(self.state) ** 2
        print('probs:', probs)

    def measure(self) -> int:
        probs = np.abs(self.state) ** 2
        outcome = np.random.choice(len(probs), p=probs / probs.sum())
        self.state[:] = 0
        self.state[outcome] = 1
        prev = load_memory() or {}
        save_memory(
            {
                "timestamp": prev.get("timestamp"),
                "ck": prev.get("ck", np.array([], dtype=np.complex64)),
                "sigma2": prev.get("sigma2", 0.0),
                "state_amplitudes": self.state,
            }
        )
        return int(outcome)

    # -------------------------------------------------------------- utilities
    def plot_bloch(self):
        """Return a matplotlib figure of the current state on the Bloch sphere."""
        if self.n != 1:
            raise ValueError("Bloch plot only available for one qubit")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        a, b = self.state
        x = 2 * (a.conjugate() * b).real
        y = 2 * (b.conjugate() * a).imag
        z = abs(a) ** 2 - abs(b) ** 2

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
        ax.plot_wireframe(
            np.cos(u) * np.sin(v),
            np.sin(u) * np.sin(v),
            np.cos(v),
            color="lightgray",
            linewidth=0.5,
        )
        ax.quiver(0, 0, 0, x, y, z, color="red", length=1.0)
        ax.set_box_aspect([1, 1, 1])
        plt.close(fig)
        return fig

    def plot_heatmap(self):
        """Return a heatmap of amplitude probabilities."""
        import matplotlib.pyplot as plt

        probs = np.abs(self.state) ** 2
        fig, ax = plt.subplots()
        ax.imshow(probs.reshape(1, -1), aspect="auto", cmap="viridis")
        ax.set_yticks([])
        ax.set_xlabel("Basis state")
        plt.close(fig)
        return fig
