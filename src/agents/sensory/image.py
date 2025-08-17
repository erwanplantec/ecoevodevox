import jax
import jax.numpy as jnp
from .base import SensoryInterface

class ImageSensoryInterface(SensoryInterface):
	"""A sensory interface that transforms the observation into a C-channel image.
	Useful for cnn based policies."""
	#-------------------------------------------------------------------
	def encode(self, obs, sensory_state):
		internal = jnp.ones((obs.internal.shape[0],*obs.chemicals.shape[1:]), obs.chemicals.dtype) * obs.internal[:,None,None]
		img = jnp.concatenate([obs.chemicals, obs.walls, internal], axis=0)
		return img, jnp.zeros((), jnp.float16), sensory_state
	#-------------------------------------------------------------------
	def init(self, policy_state, key):
		return None