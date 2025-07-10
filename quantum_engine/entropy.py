"""Positive-definite enforcement and entropy helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .geometry import logm, tr_g


def flatten(tensor: jnp.ndarray, order: str = "C") -> jnp.ndarray:
    """Flatten a tensor for spectrum computations."""
    return jnp.reshape(tensor, (-1, tensor.shape[-1])) if tensor.ndim > 1 else jnp.ravel(tensor)


def enforce_pd(matrix: jnp.ndarray, softplus: bool = True) -> jnp.ndarray:
    """Return a positive-definite version of ``matrix``."""
    vals, vecs = jnp.linalg.eigh(matrix)
    if softplus:
        vals = jax.nn.softplus(vals)
    else:
        vals = jnp.clip(vals, 1e-6)
    return (vecs * vals) @ jnp.conj(vecs.T)


def entropy_density(G: jnp.ndarray, g: jnp.ndarray) -> jnp.ndarray:
    """Diagnostic entropy density."""
    return -tr_g(logm(G @ jnp.linalg.inv(g)), g)
