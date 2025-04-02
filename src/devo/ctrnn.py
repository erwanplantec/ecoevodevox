import jax
import jax.nn as jnn
import jax.numpy as jnp
import equinox as eqx
from typing import NamedTuple, Callable

class CTRNN(NamedTuple):
    x: jax.Array
    v: jax.Array
    tau: jax.Array
    gain: jax.Array
    bias: jax.Array
    W: jax.Array
    mask: jax.Array|None=None
    m: jax.Array|None=None
    s: jax.Array|None=None
    id_: jax.Array|None=None

@eqx.filter_jit
def forward_ctrnn(ctrnn: CTRNN, I: jax.Array, dt: float=0.1, f: Callable=jnn.tanh):
    _, v, tau, gain, bias, W, mask, *_ = ctrnn
    y = f(gain*(v+bias)) * mask
    dv = - (v) + jnp.matmul(W, y) + I
    v = v + dt * (1/tau)  * dv
    return v * mask

class CTRNNPolicyConfig(NamedTuple):
    # ---
    encode_fn: Callable
    decode_fn: Callable
    activation_fn: Callable=jnn.tanh
    dt: float=0.1
    T: float=1.
    # ---

class CTRNNPolicy(eqx.Module):
    # --- CTRNN params ---
    encode: Callable
    decode: Callable
    activation_fn: Callable
    dt: float
    iters: int
    # ---
    def __init__(self, cfg: CTRNNPolicyConfig):
        self.encode = cfg.encode_fn
        self.decode = cfg.decode_fn
        self.dt = cfg.dt
        self.activation_fn = cfg.activation_fn
        self.iters = int(cfg.T // cfg.dt)
    # ---
    def __call__(self, obs: jax.Array, state: CTRNN, key: jax.Array):
        # Encode obs
        I = self.encode(obs, state)
        # forward network
        forward_fn = lambda i, state: state._replace(
            v=forward_ctrnn(state, I, dt=self.dt, f=self.activation_fn))
        state = jax.lax.fori_loop(0, self.iters, forward_fn, state)
        # decode action
        action = self.decode(state)
        return action, state
    # ---
    def initialize(self, key: jax.Array)->CTRNN:
        raise NotImplementedError()
    # ---