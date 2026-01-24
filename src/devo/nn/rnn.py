from flax.struct import PyTreeNode
import jax
import jax.nn as jnn
import jax.numpy as jnp
from typing import Callable
from jaxtyping import Float

from .core import NeuralModel, NeuralState

class RNNState(PyTreeNode):
	v: jax.Array

class RNN(NeuralModel):
	# ------------------------------------------------------------------
	W: jax.Array
	bias: jax.Array
	activation_fn: Callable
	# ------------------------------------------------------------------
	def __init__(self, 
	             nb_neurons: int=64,	            
				 activation_fn: str|Callable="relu",
				 *,
				 key: jax.Array):
		self.bias = jnp.zeros(nb_neurons)
		initializer = jnn.initializers.orthogonal()
		self.W = initializer(key, (nb_neurons, nb_neurons), dtype=jnp.float32)
		if isinstance(activation_fn, str):
			self.activation_fn = getattr(jnn, activation_fn)
		else:
			self.activation_fn = activation_fn
	# ------------------------------------------------------------------
	def __call__(self, x: jax.Array, state: RNNState, key: jax.Array)->tuple[RNNState,Float]:
		v = RNN.forward(x, state.v, self.bias, self.W, self.activation_fn)
		return state.replace(v=v), 0.0
	# ------------------------------------------------------------------
	def init(self, key: jax.Array)->RNNState:
		v = jnp.zeros(self.W.shape[0])
		return RNNState(v=v)
	# ------------------------------------------------------------------
	@classmethod
	def forward(cls, x: jax.Array, v: jax.Array, bias: jax.Array, W: jax.Array, activation_fn: Callable):
		return activation_fn(W@v + x + bias)
	# ------------------------------------------------------------------
	@classmethod
	def check_state(cls, state):
		assert hasattr(state, "v")
		assert hasattr(state, "W")
	# ------------------------------------------------------------------





