from flax.struct import PyTreeNode
import jax
import jax.nn as jnn
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, PyTree
from typing import NamedTuple, Callable

from ...eco.interface import Interface

class BasePolicy(eqx.Module):
	# --- development ---
	encoding_model: Callable[[jax.Array], PyTreeNode]
	# --- interface ---
	interface: Interface
	# ---
	def __call__(self, obs, state, key):
		obs = self.interface.encode(obs, state)
		state = self.update(obs, state, key)
		out = self.interface.decode(state)
		return out, state
	# ---
	def update(self, obs, state, key):
		raise NotImplementedError
	# ---
	def initialize(self, key: jax.Array)->PyTreeNode:
		return self.encoding_model(key)