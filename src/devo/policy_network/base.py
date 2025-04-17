from flax.struct import PyTreeNode
import jax
import jax.nn as jnn
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, PyTree
from typing import NamedTuple, Callable

class BasePolicy(eqx.Module):
	# --- development ---
	encoding_model: Callable[[jax.Array], PyTreeNode]
	# --- interface ---
	encode_fn: Callable
	decode_fn: Callable
	# ---
	def initialize(self, key: jax.Array)->PyTreeNode:
		return self.encoding_model(key)