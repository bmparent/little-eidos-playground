"""Geometry utilities for entropic action computations."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .constants import CONSTS


def logm(matrix: jnp.ndarray) -> jnp.ndarray:
    """Hermitian-safe matrix logarithm."""
    vals, vecs = jnp.linalg.eigh(matrix)
    vals = jnp.clip(vals, 1e-30)
    log_vals = jnp.log(vals)
    return (vecs * log_vals) @ jnp.conj(vecs.T)


def tr_g(A: jnp.ndarray, g: jnp.ndarray) -> jnp.ndarray:
    """Trace with respect to metric ``g``."""
    return jnp.trace(g @ A)


def entropic_action(g: jnp.ndarray, G: jnp.ndarray) -> jnp.ndarray:
    """Return the entropic action for metrics ``g`` and ``G``."""
    d = g.shape[0]
    L = -tr_g(logm(jnp.linalg.inv(G)), g)
    vol = jnp.sqrt(jnp.abs(jnp.linalg.det(-g)))
    return (1.0 / CONSTS.lP**d) * vol * L


def entropic_gradients(g: jnp.ndarray, G: jnp.ndarray):
    """Gradients of entropic action with respect to ``g`` and ``G``."""
    return jax.grad(entropic_action, argnums=(0, 1))(g, G)
