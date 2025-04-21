import jax
import jax.nn as jnn
from flax.struct import PyTreeNode
from typing import Callable, Tuple
from jaxtyping import Float

from ...eco.interface import Interface

from .base import BasePolicy

class RNN(PyTreeNode):
	v: Float[jax.Array, "N"]
	W: Float[jax.Array, "N N"]
	# ---
	mask: jax.Array

class SERNN(PyTreeNode):
	x: Float[jax.Array, "N 2"]
	v: Float[jax.Array, "N"]
	W: Float[jax.Array, "N N"]
	# ---
	mask: jax.Array


class RNNPolicy(BasePolicy):
	# ---
	activation_fn: Callable=jnn.tanh
	# ---
	def __init__(self, 
				 encoding_model: Callable[[jax.Array], RNN|SERNN],
				 interface: Interface,
				 activation_fn: Callable=jnn.tanh):
		super().__init__(encoding_model, interface)
		self.activation_fn = activation_fn
	# ---
	def update(self, obs, state: RNN|SERNN, key: jax.Array)->RNN|SERNN:
		v = self.activation_fn(state.W@state.v + obs)*state.mask
		return state.replace(v=v)
	# ---
	def initialize(self, key: jax.Array)->RNN|SERNN:
		state = self.encoding_model(key); assert isinstance(state, RNN|SERNN)
		return state
	# ---
	@classmethod
	def check_state(cls, state):
		assert hasattr(state, "v")
		assert hasattr(state, "W")




