"""Facade for optimisation loop."""

from __future__ import annotations

import jax.numpy as jnp
import optax

from .solver import EngineState, train_step
from .matter import DiracKahlerField


def create_state(d: int = 2) -> EngineState:
    g = jnp.eye(d)
    field = DiracKahlerField(jnp.zeros(()), jnp.zeros((d,)), jnp.zeros((d, d)))
    return EngineState(g=g, field=field)


def run(num_steps: int = 1_000):
    state = create_state()
    opt = optax.adam(1e-3)
    opt_state = opt.init(state)
    for _ in range(num_steps):
        state, opt_state, loss = train_step(state, opt, opt_state)
        # Simple CLI output
        if state.step % 100 == 0:
            print(f"step {state.step} loss {loss}")
    return state
