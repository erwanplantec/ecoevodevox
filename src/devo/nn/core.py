from flax.struct import PyTreeNode
import jax
import equinox as eqx
from jaxtyping import Float, PyTree
from typing import Callable, override

type NeuralState = PyTree
type NeuralInput = PyTree

class NeuralModel(eqx.Module):
	#-------------------------------------------------------------------
	def __call__(self, x: NeuralInput, state: NeuralState, key: jax.Array)->tuple[NeuralState,Float]:
		"""make a forward pass of the network
		returns updated state and energy consumption"""
		raise NotImplementedError
	#-------------------------------------------------------------------
	def init(self, key: jax.Array)->NeuralState:
		raise NotImplementedError
	#-------------------------------------------------------------------
