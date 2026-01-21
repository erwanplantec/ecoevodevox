from typing import Callable
import jax
from jax import numpy as jnp, random as jr, nn as jnn
import equinox as eqx
import equinox.nn as nn
from jaxtyping import PyTree

from jax.flatten_util import ravel_pytree

from .core import MutationModel, Genotype, PolicyParams


class GeneralizedMutation(MutationModel):
	#-------------------------------------------------------------------
	sigma: float
	p_mut: float
	num_prms: int
	reshaper: Callable
	sigma_size: float
	#-------------------------------------------------------------------
	def __init__(self, 
		sigma: float,  # Standard deviation for parameter mutations
		p_mut: float,  # Probability of mutating each parameter
		genotype_like: Genotype,  # Template genotype used to determine parameter structure
		sigma_size: float=0.0):  # Standard deviation for body size mutations
		super().__init__(sigma_size, genotype_like)
		self.sigma = sigma
		self.p_mut = p_mut
		flat_params, self.reshaper = ravel_pytree(genotype_like.policy_params)
		self.num_prms = len(flat_params)
		self.sigma_size = sigma_size
	#-------------------------------------------------------------------
	def mutate_policy_params(self, params: PolicyParams, key: jax.Array) -> PolicyParams:
		k_mut, k_locs = jr.split(key)
		epsilon = jr.normal(k_mut, (self.num_prms,)) * self.sigma
		if self.p_mut > 0.0:
			epsilon = jnp.where(
				jr.bernoulli(k_locs, self.p_mut, (self.num_prms,)),
				epsilon, 0.0
			)
		epsilon = self.reshaper(epsilon)
		return jax.tree.map(lambda x, eps: x+eps, params, epsilon)