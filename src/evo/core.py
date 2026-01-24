from typing import Callable
import jax
from jax import numpy as jnp, random as jr, nn as jnn
import equinox as eqx
import equinox.nn as nn
from jaxtyping import Float, PyTree

from ..devo.core import Genotype, NeuralParams

class MutationModel(eqx.Module):
	#-------------------------------------------------------------------
	sigma_size: float
	genotype_like: PyTree
	#-------------------------------------------------------------------
	def __call__(self, genotype: Genotype, key: jax.Array)->Genotype:
		k1, k2 = jr.split(key)
		size = self.mutate_size(genotype.body_size, k1)
		neural_params = self.mutate_neural_params(genotype.neural_params, k2)
		return Genotype(neural_params, size, genotype.chemical_emission_signature) # chemical signature is left unchanged
	#-------------------------------------------------------------------
	def mutate_size(self, size: Float, key: jax.Array)->Float:
		return size + jr.normal(key)*self.sigma_size
	#-------------------------------------------------------------------
	def mutate_neural_params(self, params: NeuralParams, key: jax.Array)->NeuralParams:
		raise NotImplementedError