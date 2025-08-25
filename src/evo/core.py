from typing import Callable
import jax
from jax import numpy as jnp, random as jr, nn as jnn
import equinox as eqx
import equinox.nn as nn
from jaxtyping import Float, PyTree

from ..agents.core import Genotype, PolicyParams

class MutationModel(eqx.Module):
	#-------------------------------------------------------------------
	sigma_size: float
	#-------------------------------------------------------------------
	def __call__(self, genotype: Genotype, key: jax.Array)->Genotype:
		k1, k2 = jr.split(key)
		size = self.mutate_size(genotype.body_size, k1)
		policy_params = self.mutate_policy_params(genotype.policy_params, k2)
		return Genotype(policy_params, size)
	#-------------------------------------------------------------------
	def mutate_size(self, size: Float, key: jax.Array)->Float:
		return size + jr.normal(key)*self.sigma_size
	#-------------------------------------------------------------------
	def mutate_policy_params(self, params: PolicyParams, key: jax.Array)->PolicyParams:
		raise NotImplementedError