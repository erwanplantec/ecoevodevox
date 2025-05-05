from flax.struct import PyTreeNode
import jax
import jax.nn as jnn
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, PyTree
from typing import NamedTuple, Callable

class NN(PyTreeNode):
	v: jax.Array
	W: jax.Array
	# ---
	mask: jax.Array

class SENN(NN):
	x: jax.Array

class Policy(eqx.Module):
	#-------------------------------------------------------------------
	encoding_model: Callable[[jax.Array], NN]
	#-------------------------------------------------------------------
	def __call__(self, obs, state, key)->tuple[NN,Float]:
		"""make a forward pass of the network
		returns updated state and energy consumption"""
		raise NotImplementedError
	#-------------------------------------------------------------------
	def init(self, key: jax.Array)->NN:
		return self.encoding_model(key)
	#-------------------------------------------------------------------