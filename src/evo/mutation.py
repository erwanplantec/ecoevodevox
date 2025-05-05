import jax
from jax import numpy as jnp, random as jr, nn as jnn
import equinox as eqx
import equinox.nn as nn
from jaxtyping import PyTree

from jax.flatten_util import ravel_pytree


def make_generalized_mutation(sigma: float, p_mut: float, *, prms_like: PyTree):
	flat_prms, reshape_epsilon_fn = ravel_pytree(prms_like)
	num_prms = len(flat_prms)
	reshaper = reshape_epsilon_fn
	#-------------------------------------------------------------------
	def _mutation_fn(prms: PyTree, key: jax.Array)->PyTree:
		k_mut, k_locs = jr.split(key)
		epsilon = jr.normal(k_mut, (num_prms,)) * sigma
		epsilon = jnp.where(
			jr.bernoulli(k_locs, p_mut, (num_prms,)),
			epsilon, 0.0
		)
		epsilon = reshaper(epsilon)
		return jax.tree.map(lambda x, eps: x+eps, prms, epsilon)
	#-------------------------------------------------------------------
	return _mutation_fn