from typing import Callable

import jax
from jax import numpy as jnp
from flax.struct import PyTreeNode
from jaxtyping import Float

from ..core import Observation, NeuralState
from .core import SensoryInterface


class RetinaSensoryState(PyTreeNode):
	"""Precomputed, per-neuron retina layout for a grown body. All fields are shape [N]:

	rows, cols:  the image pixel each neuron reads
	sensitivity: sensory-gene expression, zeroed for non-sensors
	is_sensor:   neurons expressing the sensory gene above threshold (and alive)
	"""
	rows: jax.Array
	cols: jax.Array
	sensitivity: jax.Array
	is_sensor: jax.Array
	energy_cost: Float


class RetinaSensoryInterface(SensoryInterface):
	"""Use the grown neuron layout itself as a retina.

	The image is stretched over network space [-1, 1]^2 and every *sensory* neuron reads the
	single pixel its position falls on, scaled by its sensory expression:

		I[i] = activation(s[i] * image[row(i), col(i)])   if s[i] > threshold else 0

	Images are grayscale, so the input to each neuron is a scalar. Where neurons sit
	therefore decides what the body can see: a sparse layout samples the image sparsely, and
	a clustered one is blind outside its cluster. Requires a spatially embedded (grown)
	network, i.e. a state exposing `x` and `s` (`RAND_CTRNN`, `NeuronNCA`).

	Coordinate convention (matching the rest of the codebase): `x[:, 0]` is left(-)/right(+)
	and `x[:, 1]` is back(-)/front(+). With `flip_y=True` a neuron at y=+1 reads the *top*
	row, so the retina agrees with `plt.imshow(image)` under its default `origin="upper"`.
	Neurons landing exactly on the upper bound are clipped into the edge pixel.
	"""
	#-------------------------------------------------------------------
	height: int
	width: int
	sensor_expression_threshold: float=0.03
	sensor_activation: Callable=lambda x: x
	sensor_energy_cost: float=0.0
	flip_y: bool=True
	#-------------------------------------------------------------------
	def sensory_expression(self, neural_state: NeuralState)->jax.Array:
		"""Per-neuron scalar sensitivity, masked to living neurons.

		Networks grow `s` as [N, sensory_genes] (RAND) or [N] (a single sensory gene); the
		gene axis is averaged away since a grayscale retina feeds one scalar per neuron.
		"""
		s = neural_state.s
		if s.ndim == 2:
			s = s.mean(-1)
		mask = getattr(neural_state, "mask", None)
		if mask is not None:
			s = s * mask.astype(s.dtype)
		return s
	#-------------------------------------------------------------------
	def init(self, neural_state: NeuralState, key: jax.Array)->RetinaSensoryState:
		# ---
		assert hasattr(neural_state, "x") #make sure network is spatially embedded
		assert hasattr(neural_state, "s") #make sure neurons have sensory expression
		# ---
		xs = neural_state.x
		s = self.sensory_expression(neural_state)
		is_sensor = s > self.sensor_expression_threshold

		# [-1, 1] -> [0, 1] -> pixel index; the clip catches neurons sitting on the bound
		u = (xs[:, 0] + 1.0) / 2.0
		v = (xs[:, 1] + 1.0) / 2.0
		if self.flip_y:
			v = 1.0 - v
		cols = jnp.clip(jnp.floor(u * self.width).astype(jnp.int32), 0, self.width - 1)
		rows = jnp.clip(jnp.floor(v * self.height).astype(jnp.int32), 0, self.height - 1)

		sensitivity = jnp.where(is_sensor, s, 0.0)

		return RetinaSensoryState(rows=rows,
		                          cols=cols,
		                          sensitivity=sensitivity,
		                          is_sensor=is_sensor,
		                          energy_cost=jnp.astype(self.sensor_energy_cost*sensitivity.sum(), jnp.float16))
	#-------------------------------------------------------------------
	def encode(self, obs: Observation, neural_state: NeuralState, sensory_state: RetinaSensoryState):
		"""Sample the image at each sensory neuron's pixel.

		Args:
			obs: `obs.env` is the grayscale image, [H, W] or [1, H, W].
			neural_state: the grown network; `v` supplies the shape, `s` the sensitivity
				(already baked into `sensory_state`).
			sensory_state: layout from `init`.

		Returns:
			(I, energy_cost, sensory_state, {}) with I of shape [N].
		"""
		image = obs.env
		if image.ndim == 3:
			assert image.shape[0] == 1, \
				f"retina expects a grayscale image, got {image.shape[0]} channels"
			image = image[0]
		assert image.shape == (self.height, self.width), \
			f"expected an image of shape {(self.height, self.width)}, got {image.shape}"

		pixels = image[sensory_state.rows, sensory_state.cols]
		I = jnp.where(sensory_state.is_sensor,
		              self.sensor_activation(sensory_state.sensitivity * pixels),
		              0.0)

		return I, sensory_state.energy_cost, sensory_state, {}
	#-------------------------------------------------------------------
