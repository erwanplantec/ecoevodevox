from typing import Callable
import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import equinox as eqx
import equinox.nn as nn
from jaxtyping import Float, PyTree

from ..nn.rnn import SERNN

from .base import BaseDevelopmentalModel
from .model_e import XOXT

def M(p, extra_zero_field=0):
	m = jnp.stack([p[...,0], p[...,1], jnp.abs(p[...,0]), jnp.abs(p[...,1])], axis=-1)
	if extra_zero_field:
		m = jnp.concatenate([m, jnp.zeros((*m.shape[:-1],extra_zero_field))], axis=-1)
	return m
M_dims = 4

class GRN(eqx.Module):
	Win: nn.Linear
	Wex: nn.Linear
	tau: Float
	dt: float
	has_autonomous_decay: bool
	expression_bounds: tuple[float,float]
	def __init__(self, nb_genes: int, input_dims: int, dt: float=0.03, has_autonomous_decay: bool=True,
				 expression_min: float=-jnp.inf, expression_max: float=jnp.inf,  *, key: jax.Array):
		kin, kex, kpos = jr.split(key, 3)
		self.Win = nn.Linear(nb_genes, nb_genes, key=kin, use_bias=True)
		self.Wex = nn.Linear(input_dims, nb_genes, key=kex, use_bias=True)
		self.tau = jnp.ones(nb_genes)
		self.dt = dt
		self.has_autonomous_decay = has_autonomous_decay
		self.expression_bounds = (expression_min, expression_max)
	def __call__(self, s, x):
		ds_dt = jnn.tanh(self.Win(s) + self.Wex(x))
		if self.has_autonomous_decay:
			ds_dt = ds_dt - s
		ds_dt = ds_dt / jnp.clip(self.tau, min=0.01)
		return jnp.clip(s + self.dt * ds_dt, *self.expression_bounds)

class SpatioemporalEncoder(eqx.Module):
	encoder: nn.MLP
	def __init__(self, encoding_dims: int, activation: Callable=jnn.tanh, *, key: jax.Array):
		self.encoder = nn.MLP(M_dims+1, encoding_dims, 32, 1, 
			final_activation=activation, activation=activation, key=key)
	def __call__(self, p, t):
		m = M(p)
		inp = jnp.concatenate([m,t[None]])
		output = self.encoder(inp)
		return output

class State(SERNN):
	s: jax.Array
	m: jax.Array

class GRNEncoding(BaseDevelopmentalModel):
	# ---
	grn: GRN
	encoder: SpatioemporalEncoder
	O: jax.Array
	gene_to_migration_prms: nn.Linear
	population: Float
	nb_genes: int
	population_gain: float
	T: float
	dt: float
	max_neurons: int
	genome_shaper: Callable
	extra_migration_fields: int
	# ---
	def __init__(
		self, 
		nb_sensory_genes: int, 
		nb_motor_genes: int, 
		nb_synaptic_genes: int=4, 
		nb_regulatory_genes: int=0,  
		extra_migration_fields: int=0,
		dt: float=0.03, 
		T: float=10.0, 
		population_gain: float=10., 
		max_neurons: int=128,  
		nb_init_neurons: int=8,
		*, key: jax.Array):

		grn_key, encoder_key, conn_key, g2p_key = jr.split(key, 4)

		migration_fields = M_dims + extra_migration_fields
		genes_compartments = [
			jnp.zeros(nb_sensory_genes),
			jnp.zeros(nb_motor_genes),
			jnp.zeros(nb_synaptic_genes),
			jnp.zeros(migration_fields+1),
			jnp.zeros(migration_fields),
			jnp.zeros(nb_regulatory_genes)
		]
		flat_genes, genome_shaper = ravel_pytree(genes_compartments)
		nb_genes = len(flat_genes)
		self.genome_shaper = genome_shaper
		
		self.grn = GRN(nb_genes, nb_genes, expression_max=1.0, expression_min=-1.0, has_autonomous_decay=True, key=grn_key)
		self.encoder = SpatioemporalEncoder(nb_genes, key=encoder_key)
		self.O = jr.normal(conn_key, (nb_synaptic_genes,)*2)*0.1
		self.gene_to_migration_prms = nn.Linear(migration_fields, migration_fields, use_bias=False, key=g2p_key)
		self.population = jnp.ones(())* (nb_init_neurons / population_gain) 

		self.nb_genes = nb_genes
		self.dt=dt
		self.T = T
		self.population_gain = population_gain
		self.max_neurons = max_neurons
		self.extra_migration_fields = extra_migration_fields

	# ---
	def __call__(self, key: jax.Array) -> State:

		mask = jnp.arange(self.max_neurons) < (self.population*self.population_gain)
		
		def _step(_, state):
    
			S, P, t, key = state
			key, key_noise = jr.split(key)

			X = jax.vmap(self.encoder, in_axes=(0,None))(P, t)
			S = jax.vmap(self.grn)(S, X)
			_, _, _, migration_genes, perturbation_genes, _ = jax.vmap(self.genome_shaper)(S)

			def M_(p):
				dists = jnp.sum(jnp.square(p[None]-P), axis=-1, keepdims=True)
				effect = jnp.where(mask[:,None], jnp.exp(-dists/0.1), 0.0)
				perturbations = jnp.sum(perturbation_genes * effect, axis=0)
				return M(p, self.extra_migration_fields) + perturbations

			def energy_fn(p, phi):
				m = M_(p)
				field_energy = jnp.dot(m, phi)
				return field_energy

			dP = -jax.vmap(jax.grad(energy_fn))(P, migration_genes[:,1:]) * jnp.clip(migration_genes[:,:1], 0.0, 1.0)
			dP = dP + jr.normal(key_noise, dP.shape)*0.02
			vel = jnp.linalg.norm(dP, axis=-1, keepdims=True)
			dP = jnp.where(vel>0.1, dP/vel*0.1, dP)
			dP = jnp.where(mask[:,None], dP, 0.0); assert isinstance(dP, jax.Array)
			P = jnp.clip(P+self.dt*dP, -1.0, 1.0)
			
			return [S,P,t+self.dt,key]

		P0_key, dev_key = jr.split(key, 2)

		S0 = jnp.zeros((self.max_neurons, self.nb_genes))
		P0 = jr.uniform(P0_key, (self.max_neurons, 2), minval=-0.1, maxval=0.1)

		S,P, *_ = jax.lax.fori_loop(0, int(self.T//self.dt), _step, [S0,P0,0.0,dev_key])
		
		sensory_genes, motor_genes, synaptic_genes, *_ = jax.vmap(self.genome_shaper)(S)
		W = synaptic_genes @ self.O @ synaptic_genes.T
		W = W * (mask[None]*mask[:,None])

		return State(v=jnp.zeros(self.max_neurons),W=W, mask = mask, x=P, s=sensory_genes, m=motor_genes)


#-------------------------------------------------------------------


if __name__ == '__main__':
	import matplotlib.pyplot as plt
	import numpy as np
	mdl = GRNEncoding(1, 1, 4, 8, 0, key=jr.key(np.random.randint(0,1000)))
	net = mdl(jr.key(2))
	x = net.x[net.mask]
	plt.scatter(*x.T)
	plt.xlim(-1,1.)
	plt.ylim(-1., 1.)
	plt.show()




