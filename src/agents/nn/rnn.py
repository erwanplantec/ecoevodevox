import jax
import jax.nn as jnn
from typing import Callable
from jaxtyping import Float

from .core import Policy, NN, SENN

RNN = NN
SERNN = SENN

class RNNPolicy(Policy):
	# ---
	activation_fn: Callable=jnn.tanh
	# ---
	def __init__(self, 
				 encoding_model: Callable[[jax.Array], RNN|SERNN],
				 activation_fn: Callable=jnn.tanh):
		super().__init__(encoding_model)
		self.activation_fn = activation_fn
	# ---
	def __call__(self, obs, state: RNN|SERNN, key: jax.Array)->tuple[RNN|SERNN,Float]:
		v = self.activation_fn(state.W@state.v + obs)*state.mask
		return state.replace(v=v), 0.0
	# ---
	def initialize(self, key: jax.Array)->RNN|SERNN:
		state = self.encoding_model(key); assert isinstance(state, NN|SENN)
		return state
	# ---
	@classmethod
	def check_state(cls, state):
		assert hasattr(state, "v")
		assert hasattr(state, "W")




