"""Quantum computation core for little-eidos-playground.

This module implements a minimal multi-qubit engine with common gates and
measurement utilities.  It is backwards compatible with the previous
``QuantumToy`` API via ``run``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class QuantumEngine:
    """Simple multi-qubit quantum state simulator."""

    n_qubits: int = 2

    def __post_init__(self) -> None:
        self.n = int(self.n_qubits)
        self.state = np.zeros(2 ** self.n, dtype=complex)
        self.state[0] = 1.0

    # ------------------------------------------------------------------
    # Gate helpers
    # ------------------------------------------------------------------
    def _single_qubit_matrix(self, M: np.ndarray, qubit: int) -> np.ndarray:
        eye_left = np.eye(2 ** qubit, dtype=complex)
        eye_right = np.eye(2 ** (self.n - qubit - 1), dtype=complex)
        return np.kron(np.kron(eye_left, M), eye_right)

    def _two_qubit_matrix(self, M: np.ndarray, control: int, target: int) -> np.ndarray:
        qubits = sorted([control, target])
        eye_before = np.eye(2 ** qubits[0], dtype=complex)
        eye_between = np.eye(2 ** (qubits[1] - qubits[0] - 1), dtype=complex)
        eye_after = np.eye(2 ** (self.n - qubits[1] - 1), dtype=complex)
        mat = np.kron(np.kron(np.kron(eye_before, M), eye_between), eye_after)
        if control > target:
            # swap to apply on correct qubits
            swap = self._swap_matrix(control, target)
            mat = swap @ mat @ swap
        return mat

    def _swap_matrix(self, q1: int, q2: int) -> np.ndarray:
        I = np.eye(2, dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        if q1 == q2:
            return np.eye(2 ** self.n, dtype=complex)
        mats = []
        for q in range(self.n):
            if q == q1 or q == q2:
                mats.append(X)
            else:
                mats.append(I)
        res = mats[0]
        for m in mats[1:]:
            res = np.kron(res, m)
        return res

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def apply(self, gate: str, *qubits: int, theta: float | None = None) -> None:
        """Apply a gate to the register."""

        gate = gate.upper()
        if gate == "I":
            M = np.eye(2, dtype=complex)
            U = self._single_qubit_matrix(M, qubits[0])
        elif gate == "X":
            M = np.array([[0, 1], [1, 0]], dtype=complex)
            U = self._single_qubit_matrix(M, qubits[0])
        elif gate == "Y":
            M = np.array([[0, -1j], [1j, 0]], dtype=complex)
            U = self._single_qubit_matrix(M, qubits[0])
        elif gate == "Z":
            M = np.array([[1, 0], [0, -1]], dtype=complex)
            U = self._single_qubit_matrix(M, qubits[0])
        elif gate == "H":
            M = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
            U = self._single_qubit_matrix(M, qubit=qubits[0])
        elif gate == "S":
            M = np.array([[1, 0], [0, 1j]], dtype=complex)
            U = self._single_qubit_matrix(M, qubits[0])
        elif gate == "T":
            M = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
            U = self._single_qubit_matrix(M, qubits[0])
        elif gate in {"RX", "RY", "RZ"}:
            assert theta is not None, "Rotation gates require theta"
            if gate == "RX":
                M = np.array(
                    [
                        [np.cos(theta / 2), -1j * np.sin(theta / 2)],
                        [-1j * np.sin(theta / 2), np.cos(theta / 2)],
                    ],
                    dtype=complex,
                )
            elif gate == "RY":
                M = np.array(
                    [
                        [np.cos(theta / 2), -np.sin(theta / 2)],
                        [np.sin(theta / 2), np.cos(theta / 2)],
                    ],
                    dtype=complex,
                )
            else:  # RZ
                M = np.array(
                    [
                        [np.exp(-1j * theta / 2), 0],
                        [0, np.exp(1j * theta / 2)],
                    ],
                    dtype=complex,
                )
            U = self._single_qubit_matrix(M, qubits[0])
        elif gate in {"CNOT", "CZ"}:
            control, target = qubits
            if gate == "CNOT":
                M = np.array(
                    [
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, 1, 0],
                    ],
                    dtype=complex,
                )
            else:  # CZ
                M = np.diag([1, 1, 1, -1]).astype(complex)
            U = self._two_qubit_matrix(M, control, target)
        else:
            raise ValueError(f"Unknown gate {gate}")

        self.state = U @ self.state

    def measure(self) -> str:
        """Collapse the state and return a bitstring."""
        probs = np.abs(self.state) ** 2
        idx = np.random.choice(len(self.state), p=probs)
        self.state = np.zeros_like(self.state)
        self.state[idx] = 1
        return format(idx, f"0{self.n}b")

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------
    def plot_bloch(self, qubit: int = 0) -> None:
        if self.n != 1:
            raise ValueError("Bloch sphere is defined for single qubit only")
        a, b = self.state
        theta = 2 * np.arccos(np.abs(a))
        phi = np.angle(b) - np.angle(a)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.quiver(0, 0, 0, x, y, z, length=1.0)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        plt.show()

    def plot_heatmap(self) -> None:
        amplitudes = np.abs(self.state.reshape(-1, 1))
        plt.imshow(amplitudes, cmap="viridis")
        plt.colorbar(label="Amplitude")
        plt.xlabel("state index")
        plt.show()

    # ------------------------------------------------------------------
    # Backwards compatible API
    # ------------------------------------------------------------------
    def entropy(self, level: float) -> None:
        phase = np.exp(1j * np.random.uniform(-level, level))
        self.state *= phase

    def apply_gate(self, gate_name: str) -> None:  # compatibility shim
        self.apply(gate_name, 0)

    def run(self, glyphs: Iterable[str]) -> None:
        mapping = {"₿": "X", "☀️": "H", "📈": "I"}
        for g in glyphs:
            gate = mapping.get(g)
            if gate:
                self.apply(gate, 0)

