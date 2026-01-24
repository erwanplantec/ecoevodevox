import jax
import jax.numpy as jnp
from jaxtyping import Float, Float16

from devo.core import SensoryState
from .core import SensoryInterface, NeuralState

class Image: Float[jax.Array, "C H W"]

class ImageSensoryInterface(SensoryInterface):
	"""A sensory interface that transforms the observation into a C-channel image.
	Useful for cnn based policies."""
	#-------------------------------------------------------------------
	def encode(self, obs, neural_state: NeuralState, sensory_state: None)->tuple[Image, Float16,  SensoryState, dict]:
		_, H, W = obs.env.shape
		internal_im = jnp.tile(obs.internal[:,None,None], (1, H, W))
		img = jnp.concatenate([obs.env, internal_im], axis=0); assert isinstance(img, Image)
		return img, jnp.zeros((), dtype=jnp.float16), None, {}
	#-------------------------------------------------------------------
	def init(self, neural_state: NeuralState, key: jax.Array):
		return None