"""Training loop utilities for the quantum engine."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax
import equinox as eqx

from .geometry import entropic_action
from .induced_metric import assemble_G
from .entropy import enforce_pd
from .matter import DiracKahlerField


@dataclass
class EngineState:
    g: jnp.ndarray
    field: DiracKahlerField
    step: int = 0

    @property
    def G(self) -> jnp.ndarray:
        return assemble_G(self.g, self.field)


def physics_loss(state: EngineState) -> jnp.ndarray:
    G = enforce_pd(state.G)
    return jnp.mean(entropic_action(state.g, G))


def train_step(state: EngineState, opt: optax.GradientTransformation, opt_state):
    loss, grads = jax.value_and_grad(physics_loss)(state)
    updates, opt_state = opt.update(grads, opt_state)
    state = eqx.apply_updates(state, updates)
    state.step += 1
    return state, opt_state, loss
