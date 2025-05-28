import jax
import jax.nn as jnn
import jax.numpy as jnp
import equinox as eqx
from typing import Callable

from .base import Policy, NN, SENN

class CTRNN(NN):
    tau: jax.Array
    gain: jax.Array

class SECTRNN(SENN):
    tau: jax.Array
    gain: jax.Array

@eqx.filter_jit
def forward_ctrnn(ctrnn: CTRNN, I: jax.Array, dt: float=0.1, f: Callable=jnn.tanh):
    y  = f(ctrnn.gain*(ctrnn.v+ctrnn.b)) * ctrnn.mask
    dv = - (ctrnn.v) + jnp.matmul(ctrnn.W, y) + I
    v  = ctrnn.v + dt * (1/ctrnn.tau)  * dv
    v = jnp.clip(v, -5.0, 5.0)
    return v * ctrnn.mask

class CTRNNPolicy(Policy):
    # --- CTRNN params ---
    activation_fn: Callable
    dt: float
    iters: int
    # ---
    def __init__(self, 
                 encoding_model, 
                 dt=0.1, 
                 T=1.0, 
                 activation_fn=jnn.tanh):
        super().__init__(encoding_model)
        self.activation_fn = activation_fn
        self.dt = dt
        self.iters = int(T//dt)
    # ---
    def __call__(self, obs: jax.Array, state: CTRNN|SECTRNN, key: jax.Array):
        # forward network
        forward_fn = lambda i, state: state.replace(
            v=forward_ctrnn(state, obs, dt=self.dt, f=self.activation_fn))
        state = jax.lax.fori_loop(0, self.iters, forward_fn, state)
        return state, 0.0
    # ---
    def initialize(self, key: jax.Array)->CTRNN|SECTRNN:
        state = self.encoding_model(key); assert isinstance(state, CTRNN|SECTRNN)
        return state
    # ---