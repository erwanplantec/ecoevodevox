import jax
import jax.numpy as jnp
import jax.random as jr

from .base import DevelopmentalModel
from src.agents.nn import CTRNNPolicy, CTRNN, RNN

class DirectCTRNN(DevelopmentalModel):
    # ---
    tau: jax.Array
    W: jax.Array
    b: jax.Array
    gain: jax.Array
    # ---
    def __init__(self, nb_neurons: int, *, key: jax.Array):
        """
        Direct encoding for CTRNN.
        """
        key, key_tau, key_W, key_b, key_gain = jr.split(key, 5)
        self.tau = jr.normal(key_tau, (nb_neurons,))
        self.W = jr.normal(key_W, (nb_neurons, nb_neurons))
        self.b = jr.normal(key_b, (nb_neurons,))
        self.gain = jr.normal(key_gain, (nb_neurons,))
    # ---
    def __call__(self, key: jax.Array|None=None)->CTRNN:
        return CTRNN(tau=self.tau, W=self.W, b=self.b, gain=self.gain, v=jnp.zeros(self.W.shape[0]), mask=jnp.ones(self.W.shape[0], dtype=bool))

class DirectRNN(DevelopmentalModel):
    # ---
    W: jax.Array
    b: jax.Array
    # ---
    def __init__(self, nb_neurons: int, *, key: jax.Array):
        key_W, key_b = jr.split(key, 2)
        self.W = jr.normal(key_W, (nb_neurons, nb_neurons))
        self.b = jr.normal(key_b, (nb_neurons,))
    # ---
    def __call__(self, key: jax.Array|None=None)->RNN:
        return RNN(W=self.W, b=self.b, v=jnp.zeros(self.W.shape[0]), mask=jnp.ones(self.W.shape[0], dtype=bool))