import jax
import jax.numpy as jnp

from .base import SensoryInterface

def get_edge_values(x: jax.Array):
	return jnp.concatenate(
       [jnp.ravel(x[:,0,:]), jnp.ravel(x[:,1:,-1]), jnp.ravel(x[:,-1,:-1]), jnp.ravel(x[:,1:-1,0])]
    )

class FlattenSensoryInterface(SensoryInterface):
	"""A sensory interface that flattens the sensory input."""
	#-------------------------------------------------------------------
	subset: str="all"
	#-------------------------------------------------------------------
	def encode(self, obs, policy_state, sensory_state):
		n = policy_state.v.shape[0]
		walls_and_chems = jnp.concatenate([obs.walls, obs.chemicals], axis=0)
		if self.subset=="all":
			o = jnp.concatenate([jnp.ravel(walls_and_chems), jnp.ravel(obs.internal)], axis=0)
		elif self.subset=="edges":
			o = get_edge_values(walls_and_chems)
			o = jnp.concatenate([o, obs.internal])
		elif self.subset=="front":
			o = jnp.ravel(walls_and_chems[:,:,-1])
			o = jnp.concatenate([o, obs.internal])
		else:
			raise ValueError(f"subset {self.subset} is not valid")
		inp = jnp.zeros(n).at[:len(o)].set(o)
		return inp, jnp.zeros((), jnp.float16), sensory_state, {}
	#-------------------------------------------------------------------
	def init(self, policy_state, key):
		return None

