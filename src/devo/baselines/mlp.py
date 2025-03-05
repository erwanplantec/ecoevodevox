import equinox as eqx
import equinox.nn as nn
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp

class MLPPolicy(eqx.Module):
	# ---
	mlp: nn.MLP
	# ---
	def __init__(self, obs_dims, n_actions=6, width=32, depth=2, *, key):
		self.mlp = nn.MLP(obs_dims, n_actions, width_size=width, depth=depth, key=key)
	# ---
	def __call__(self, obs, state, key):
		obs, _ = ravel_pytree(obs)
		y = self.mlp(obs) #type:ignore
		return jnp.argmax(y), state

	# ---
	def initialize(self, key):
		return None