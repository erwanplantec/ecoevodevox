import jax
import jax.numpy as jnp
from .core import SensoryInterface, PolicyState

class ImageSensoryInterface(SensoryInterface):
	"""A sensory interface that transforms the observation into a C-channel image.
	Useful for cnn based policies."""
	#-------------------------------------------------------------------
	def encode(self, obs, policy_state: PolicyState, sensory_state: None):
		internal = jnp.ones((obs.internal.shape[0],*obs.chemicals.shape[1:]), obs.chemicals.dtype) * obs.internal[:,None,None]
		img = jnp.concatenate([obs.chemicals, obs.walls, internal], axis=0)
		return img, jnp.zeros((), jnp.float16), sensory_state, {} 
	#-------------------------------------------------------------------
	def init(self, policy_state: PolicyState, key: jax.Array):
		return None