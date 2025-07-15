"""Curiosity helpers."""

from pathlib import Path
from typing import Sequence, Union
import numpy as np

NumberArray = Union[Sequence[float], np.ndarray]

STORE = Path("metrics/vector_store.npy")
STORE.parent.mkdir(exist_ok=True)


def _as_float_array(vec: Union[NumberArray, bytes, str]) -> np.ndarray:
    """Return ``vec`` as ``np.float32`` array, raising ``ValueError`` for bytes."""
    if isinstance(vec, (bytes, str)):
        raise ValueError(
            "Input vector must be a numeric array, not bytes or string."
        )
    return np.asarray(vec, dtype=np.float32)


def calc_surprise(vector: Union[NumberArray, bytes, str]) -> float:
    """Return cosine distance of ``vector`` to the centroid of the store."""
    vector = _as_float_array(vector)
    if vector.ndim != 1:
        raise ValueError("vector must be 1-D")
    if STORE.exists():
        data = np.load(STORE)
        centroid = data.mean(axis=0)
        data = np.vstack([data, vector])
    else:
        centroid = np.zeros_like(vector)
        data = vector[None, :]
    denom = (np.linalg.norm(vector) * np.linalg.norm(centroid)) + 1e-8
    surprise = 1.0 - float(np.dot(vector, centroid) / denom) if denom else 0.0
    np.save(STORE, data)
    return surprise
