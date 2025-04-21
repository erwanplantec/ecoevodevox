import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import equinox as eqx
import equinox.nn as nn

from .base import BaseDevelopmentalModel

class GRN(eqx.Module):
	Win: nn.Linear
	Wex: nn.Linear
	dt: float
	def __init__(self, nb_genes: int, input_dims: int, dt: float=0.03, *, key: jax.Array):
		kin, kex, kpos = jr.split(key, 3)
		self.Win = nn.Linear(nb_genes, nb_genes, key=kin, use_bias=True)
		self.Wex = nn.Linear(input_dims, nb_genes, key=kex, use_bias=True)
		self.dt = dt
	def __call__(self, s, x):
		ds_dt = jnn.tanh(self.Win(s) + self.Wex(x))
		return jnp.clip(s + self.dt * ds_dt, -1.0, 1.0)

class GRNEncoding(BaseDevelopmentalModel):
	# ---
	grn: GRN
	# ---
	def __init__(self, nb_genes: int):

		self.grn = 