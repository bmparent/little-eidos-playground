"""Dirac--K\xE4hler matter field stack."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass
class DiracKahlerField:
    scalar: jnp.ndarray
    one_form: jnp.ndarray
    two_form: jnp.ndarray

    def d(self) -> "DiracKahlerField":
        """Exterior derivative stencil (placeholder)."""
        # TODO: implement finite difference operators
        return self  # placeholder

    def delta(self) -> "DiracKahlerField":
        """Coderivative stencil (placeholder)."""
        # TODO: implement coderivative
        return self  # placeholder

    def kinetic_block(self, g: jnp.ndarray) -> jnp.ndarray:
        """Return matter current pieces for induced metric."""
        # TODO: compute gradients and assemble tensor
        return jnp.zeros_like(g)
