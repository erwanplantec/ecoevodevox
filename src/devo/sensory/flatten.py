from typing import Literal
import jax
import jax.numpy as jnp

from .core import SensoryInterface

def get_edge_values(x: jax.Array):
	return jnp.concatenate(
       [jnp.ravel(x[:,0,:]), jnp.ravel(x[:,1:,-1]), jnp.ravel(x[:,-1,:-1]), jnp.ravel(x[:,1:-1,0])]
    )

class FlattenSensoryInterface(SensoryInterface):
	"""A sensory interface that flattens the sensory input.
	1. take a subset of env inputs 
	2. flatten it
	3. concat with internals
	4. pad to nn size"""
	#-------------------------------------------------------------------
	subset: Literal["all", "edges", "front"]="all"
	#-------------------------------------------------------------------
	def encode(self, obs, neural_state, sensory_state):
		n = neural_state.v.shape[0]
		if self.subset=="all":
			o = jnp.concatenate([jnp.ravel(obs.env), obs.internal], axis=0)
		elif self.subset=="edges":
			o = get_edge_values(obs.env)
			o = jnp.concatenate([o, obs.internal])
		elif self.subset=="front":
			o = jnp.ravel(obs.env[:,-1])
			o = jnp.concatenate([o, obs.internal])
		else:
			raise ValueError(f"subset {self.subset} is not valid")
		inp = jnp.zeros(n).at[:len(o)].set(o)
		return inp, jnp.zeros((), jnp.float16), sensory_state, {}
	#-------------------------------------------------------------------
	def init(self, neural_state, key):
		return None

