"""Curiosity helpers."""

from pathlib import Path
import numpy as np

STORE = Path("metrics/vector_store.npy")
STORE.parent.mkdir(exist_ok=True)


def calc_surprise(vector: np.ndarray) -> float:
    """Return cosine distance of ``vector`` to the centroid of the store."""
    vector = np.asarray(vector, dtype=np.float32)
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
