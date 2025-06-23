"""Simple quantum engine supporting up to four qubits."""

import math
import random
from typing import List

import numpy as np

from state import load_state, save_state


def _single_qubit_gate(gate: np.ndarray, n: int, target: int) -> np.ndarray:
    """Construct a full gate matrix acting on ``target`` qubit among ``n``."""
    result = np.array([[1]], dtype=np.complex64)
    for i in range(n):
        if i == target:
            result = np.kron(result, gate)
        else:
            result = np.kron(result, np.eye(2, dtype=np.complex64))
    return result


def _two_qubit_kron(M: np.ndarray, n: int, control: int, target: int) -> np.ndarray:
    """Return full matrix for a 2-qubit gate acting on ``control`` and ``target``."""
    dim = 2**n
    full = np.zeros((dim, dim), dtype=np.complex64)
    for i in range(dim):
        vec = np.zeros(dim, dtype=np.complex64)
        vec[i] = 1
        arr = vec.reshape([2] * n)
        arr = np.moveaxis(arr, [control, target], [n - 2, n - 1])
        arr = arr.reshape(-1, 4) @ M.T
        arr = arr.reshape([2] * n)
        arr = np.moveaxis(arr, [n - 2, n - 1], [control, target])
        full[:, i] = arr.reshape(-1)
    return full


class QuantumToy:
    """Lightweight quantum simulator."""

    def __init__(self, n: int = 1):
        self.n = n
        dim = 2**n
        self.state = np.zeros(dim, dtype=np.complex64)
        self.state[0] = 1 + 0j
        mem = load_state()
        if mem is not None:
            amps = mem.get("qubit_state")
            if isinstance(amps, np.ndarray) and amps.shape == (dim,):
                self.state = amps

    # ------------------------------------------------------------------ gates
    def entropy(self, level: float) -> None:
        phase = np.exp(1j * random.uniform(-level, level))
        self.state *= phase

    def apply(self, gate: str, *qubits: int, theta: float | None = None) -> None:
        single = {
            "X": np.array([[0, 1], [1, 0]], dtype=np.complex64),
            "H": (1 / math.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex64),
            "I": np.eye(2, dtype=np.complex64),
        }
        two = {
            "CNOT": np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
                dtype=np.complex64,
            ),
            "CZ": np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]],
                dtype=np.complex64,
            ),
            "SWAP": np.array(
                [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                dtype=np.complex64,
            ),
        }

        if gate in {"RZ", "RY"}:
            if theta is None:
                raise ValueError(f"{gate} gate requires theta")
            if gate == "RZ":
                single["RZ"] = np.array(
                    [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]],
                    dtype=np.complex64,
                )
            else:
                single["RY"] = np.array(
                    [
                        [np.cos(theta / 2), -np.sin(theta / 2)],
                        [np.sin(theta / 2), np.cos(theta / 2)],
                    ],
                    dtype=np.complex64,
                )

        if gate in single:
            q = qubits[0] if qubits else 0
            full = _single_qubit_gate(single[gate], self.n, q)
        elif gate in two:
            if len(qubits) != 2:
                raise ValueError("Two-qubit gate requires two qubit indices")
            full = _two_qubit_kron(two[gate], self.n, qubits[0], qubits[1])
        else:
            raise ValueError(f"Unknown gate {gate}")

        self.state = full @ self.state
        norm = np.linalg.norm(self.state)
        if not np.isclose(norm, 1.0, atol=1e-6):
            raise ValueError(f"State norm drift {norm}")

    def apply_gate(
        self, gate_name: str, qubit: int = 0, theta: float | None = None
    ) -> None:
        if gate_name in {"CNOT", "CZ", "SWAP"}:
            raise ValueError("use apply() for multi-qubit gates")
        self.apply(gate_name, qubit, theta=theta)

    def show_probabilities(self) -> None:
        probs = np.abs(self.state) ** 2
        print("probs:", probs)

    def measure(self) -> int:
        probs = np.abs(self.state) ** 2
        outcome = np.random.choice(len(probs), p=probs / probs.sum())
        self.state[:] = 0
        self.state[outcome] = 1
        state = load_state()
        state["qubit_state"] = self.state
        save_state(state)
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
        if self.n % 2 == 0:
            side = 2 ** (self.n // 2)
            arr = probs.reshape(side, side)
        else:
            arr = probs.reshape(1, -1)
        fig, ax = plt.subplots()
        ax.imshow(arr, aspect="auto", cmap="viridis")
        ax.set_yticks([])
        ax.set_xlabel("Basis state")
        plt.close(fig)
        return fig
