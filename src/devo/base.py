from flax.struct import PyTreeNode
import jax
import equinox as eqx
from jaxtyping import PyTree


class BaseDevelopmentalModel(eqx.Module):

	def __call__(self, key: jax.Array)->PyTreeNode:
		raise NotImplementedError
