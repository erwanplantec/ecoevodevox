from flax.struct import PyTreeNode
import jax
import equinox as eqx
from jaxtyping import Float, PyTree
from typing import Callable

class NN(PyTreeNode):
	v: jax.Array
	W: jax.Array
	b: jax.Array
	# ---
	mask: jax.Array

class SENN(NN):
	x: jax.Array

type PolicyState=PyTree

class Policy(eqx.Module):
	#-------------------------------------------------------------------
	encoding_model: Callable[[jax.Array], PolicyState]
	#-------------------------------------------------------------------
	def __call__(self, obs: PyTree, state: PolicyState, key: jax.Array)->tuple[PolicyState,Float]:
		"""make a forward pass of the network
		returns updated state and energy consumption"""
		raise NotImplementedError
	#-------------------------------------------------------------------
	def init(self, key: jax.Array)->PolicyState:
		return self.encoding_model(key)
	#-------------------------------------------------------------------