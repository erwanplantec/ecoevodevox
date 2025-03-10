import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import equinox as eqx
import equinox.nn as nn
from typing import NamedTuple

class NDPState(NamedTuple):
	h: jax.Array
	m: jax.Array
	A: jax.Array


class ModularNDP(eqx.Module):
	"""
	"""
	#-------------------------------------------------------------------
	# Parameters:
	update_fn: nn.MLP
	message_fn: nn.MLP
	# Statics:
	
	#-------------------------------------------------------------------

	def __init__(self, hidden_dims: int, *, key: jax.Array):
		
		k1, k2 = jr.split(key)
		self.update_fn = nn.MLP(hidden_dims*2, hidden_dims, 64, 1, key=k1)
		self.message_fn = nn.MLP(hidden_dims, hidden_dims, 64, 1, key=k2)

	#-------------------------------------------------------------------
	def __call__(self, obs, state, key):
		pass
	#-------------------------------------------------------------------
	def initialize(self, key)->NDPState:
		pass