import jax
import jax.numpy as jnp

from .base import SensoryInterface

class FlattenSensoryInterface(SensoryInterface):
	"""A sensory interface that flattens the sensory input."""
	#-------------------------------------------------------------------
	def encode(self, obs, sensory_state):
		return jnp.concatenate([jnp.ravel(obs.chemicals), jnp.ravel(obs.walls), jnp.ravel(obs.internal)], axis=0), jnp.zeros((), jnp.float16), sensory_state
	#-------------------------------------------------------------------
	def init(self, policy_state, key):
		return None