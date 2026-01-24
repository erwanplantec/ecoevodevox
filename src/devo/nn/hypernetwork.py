from typing import Callable
from flax.struct import PyTreeNode
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import equinox.nn as nn
from jaxtyping import Float

from .rnn import RNN
from .ctrnn import CTRNN
from .core import NeuralModel


class Hypernetwork(NeuralModel):
    # ------------------------------------------------------------------
    network: nn.Sequential
    sigma: float|Float
    latent_dims: int
    # ------------------------------------------------------------------
    def __init__(self, latent_dims: int=2, activation_fn: str|Callable="tanh", final_activation_fn: str="id", sigma: float=0.1, sigma_is_parameter: bool=False, *, key: jax.Array):
        
        self.latent_dims = latent_dims

        act_fn = getattr(jnn, activation_fn) if isinstance(activation_fn, "str") else activation_fn
        final_act_fn = getattr(jnn, final_activation_fn) if isinstance(final_activation_fn, "str") else final_activation_fn

        k1, k2, k3, k4, k5, k6, k7 = jr.split(key, 7)

        self.network = nn.Sequential([
            nn.Linear(latent_dims, 64, key=k1),
            lambda x, key: act_fn(x),
            lambda x, key: x[:,None,None],
            nn.ConvTranspose2d(64, 32, 2, 2, key=k2), #2
            lambda x, key: act_fn(x),
            nn.ConvTranspose2d(32, 16, 2, 2, key=k3), #4
            lambda x, key: act_fn(x),
            nn.ConvTranspose2d(16, 8, 2, 2, key=k4), #8
            lambda x, key: act_fn(x),
            nn.ConvTranspose2d(8, 4, 2, 2, key=k5), #16
            lambda x, key: act_fn(x),
            nn.ConvTranspose2d(4, 2, 2, 2, key=k6), #32
            lambda x, key: act_fn(x),
            nn.ConvTranspose2d(2, 1, 2, 2, key=k7), #64
            lambda x, key: final_act_fn(x)
        ])

        self.sigma = jnp.ones(())*sigma if sigma_is_parameter else sigma
        self.bias = jnp.zeros((64,))
    # ------------------------------------------------------------------
    def sample_latent(self, key: jax.Array):
        return jr.normal(key, (self.latent_dims,)) * self.sigma

class HyperRNNState(PyTreeNode):
    v: jax.Array
    W: jax.Array
    bias: jax.Array

class HyperRNN(Hypernetwork):
    # ------------------------------------------------------------------
    bias: jax.Array
    rnn_activation_fn: Callable
    # ------------------------------------------------------------------
    def __init__(self, rnn_activation: str|Callable="relu", latent_dims: int=2, activation_fn: str="tanh", final_activation_fn: str="id", sigma: float=0.1, sigma_is_parameter: bool=False, *, key: jax.Array):
        super().__init__(latent_dims, activation_fn, final_activation_fn, sigma, sigma_is_parameter, key=key)
        self.bias = jnp.zeros(64)
        if isinstance(rnn_activation, str):
            self.rnn_activation_fn = getattr(jnn, rnn_activation)
        else:
            self.rnn_activation_fn = rnn_activation
    # ------------------------------------------------------------------
    def __call__(self, x: jax.Array, state: HyperRNNState, key: jax.Array) -> tuple[HyperRNNState, Float]:
        return RNN.forward(x, state.v, state.bias, state.W, self.rnn_activation_fn), 0.0
    # ------------------------------------------------------------------
    def init(self, key: jax.Array) -> HyperRNNState:
        z = self.sample_latent(key)
        W = self.network(z)[0]
        n, *_ = W.shape
        return HyperRNNState(v=jnp.zeros(W.shape[0]), W=W, bias=self.bias)
    # ------------------------------------------------------------------