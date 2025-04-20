import jax
import jax.nn as jnn
import jax.numpy as jnp
import equinox as eqx
from typing import Callable
from flax.struct import PyTreeNode

from .base import BasePolicy
from .rnn import RNN, SERNN

class CTRNN(RNN):
    tau: jax.Array
    gain: jax.Array
    bias: jax.Array

class SECTRNN(SERNN):
    tau: jax.Array
    gain: jax.Array
    bias: jax.Array

@eqx.filter_jit
def forward_ctrnn(ctrnn: CTRNN, I: jax.Array, dt: float=0.1, f: Callable=jnn.tanh):
    y  = f(ctrnn.gain*(ctrnn.v+ctrnn.bias)) * ctrnn.mask
    dv = - (ctrnn.v) + jnp.matmul(ctrnn.W, y) + I
    v  = ctrnn.v + dt * (1/ctrnn.tau)  * dv
    return v * ctrnn.mask

class CTRNNPolicy(BasePolicy):
    # --- CTRNN params ---
    activation_fn: Callable
    dt: float
    iters: int
    # ---
    def __init__(self, 
                 encoding_model, 
                 encode_fn, 
                 decode_fn, 
                 dt=0.1, 
                 T=1.0, 
                 activation_fn=jnn.tanh):
        super().__init__(encoding_model, encode_fn, decode_fn)
        self.activation_fn = activation_fn
        self.dt = dt
        self.iters = int(T//dt)
    # ---
    def __call__(self, obs: jax.Array, state: CTRNN|SECTRNN, key: jax.Array):
        # Encode obs
        I = self.encode_fn(obs, state)
        # forward network
        forward_fn = lambda i, state: state.replace(
            v=forward_ctrnn(state, I, dt=self.dt, f=self.activation_fn))
        state = jax.lax.fori_loop(0, self.iters, forward_fn, state)
        # decode action
        action = self.decode_fn(state)
        return action, state
    # ---
    def initialize(self, key: jax.Array)->CTRNN|SECTRNN:
        state = self.encoding_model(key); assert isinstance(state, CTRNN|SECTRNN)
        return state
    # ---