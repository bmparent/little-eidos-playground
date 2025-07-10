"""Assemble the induced metric G."""

from __future__ import annotations

import jax.numpy as jnp

from .constants import CONSTS
from .matter import DiracKahlerField


def compute_ricci(g: jnp.ndarray) -> jnp.ndarray:
    """Placeholder Ricci tensor computation."""
    # TODO: implement finite element Ricci tensor
    return jnp.zeros_like(g)


def assemble_G(g: jnp.ndarray, field: DiracKahlerField) -> jnp.ndarray:
    """Return the induced metric G as in Eq (2)."""
    d = g.shape[0]
    alpha = CONSTS.alpha0 * CONSTS.lP**d
    beta = CONSTS.beta0 * CONSTS.lP**2
    M = field.kinetic_block(g)
    R = compute_ricci(g)
    return g + alpha * M - beta * R
