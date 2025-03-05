import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
from jaxtyping import Float

def mutate_flat_generalized(prms: jax.Array, key: jax.Array, sigma: Float, p: Float):
	"""With probability p, mutates parameters with gaussian noise with std sigma"""
	kmut, keps = jr.split(key)
	epsilon = jr.normal(keps, prms.shape, prms.dtype) * sigma
	mask = jr.bernoulli(kmut, p, prms.shape)
	return jnp.where(mask, prms+epsilon, prms)