from ..core import Observation, NeuralState
from .core import SensoryInterface

import jax
from jax import numpy as jnp, random as jr, nn as jnn
import equinox.nn as nn
from typing import Callable
from flax.struct import PyTreeNode
from jaxtyping import Bool, Float

class State(PyTreeNode):
	"""A state for a spatially embedded sensory interface."""
	on_border: Bool
	s: jax.Array
	mask: jax.Array
	indices: jax.Array
	energy_cost: Float

class SpatiallyEmbeddedSensoryInterface(SensoryInterface):
	"""A sensory interface that processes spatial information from the environment.
	
	This class implements a spatially embedded sensory system that processes chemical,
	wall, and internal signals from the environment. It uses a threshold-based activation
	system to determine which sensors are active and processes their inputs based on
	their spatial positions.
	
	Attributes:
		sensor_expression_threshold (float): Minimum threshold for sensor activation.
		sensor_activation (Callable): Activation function for sensor outputs (default: tanh).
		border_threshold (float): Threshold for determining if a neuron is on the border.
	"""
	#-------------------------------------------------------------------
	body_resolution: int
	sensor_expression_threshold: float=0.03
	sensor_activation: Callable=lambda x: x
	border_threshold: float=0.0
	sensor_energy_cost: float=0.0
	#-------------------------------------------------------------------
	def sensory_expression(self, neural_state)->jax.Array:
		"""Process the sensory expression from the policy state.
		
		Args:
			neural_state: An object containing sensor states (s) and masks.
			
		Returns:
			jnp.ndarray: Processed sensory signals after thresholding and activation.
		"""
		s = neural_state.s * neural_state.mask[:,None]
		s = jnp.where(s>self.sensor_expression_threshold, s, 0.0); assert isinstance(s, jax.Array)
		return s
	#-------------------------------------------------------------------
	def init(self, neural_state, key):
		# ---
		assert hasattr(neural_state, "x") #make sure network is spatially embedded
		assert hasattr(neural_state, "v") #make sure neurons have activation
		assert hasattr(neural_state, "s") #make sure neurons have sensory expression
		# ---
		xs = neural_state.x
		s = self.sensory_expression(neural_state)
		on_border = jnp.any(jnp.abs(xs)>self.border_threshold, axis=-1) #check if neuron is on border (epithelial layer)
		_xs = (xs+1)/2.0001 #make sure it does not reach upper bound
		coords = jnp.floor(_xs * self.body_resolution)
		coords = coords.astype(jnp.int16)

		return State(on_border=on_border, 
			   		 s=s, 
			  	 	 mask=neural_state.mask, 
					 indices=coords, 	
					 energy_cost=jnp.astype(self.sensor_energy_cost*s.sum(), jnp.float16))
	#-------------------------------------------------------------------
	def encode(self, obs: Observation, neural_state: NeuralState, sensory_state: State):
		"""Encode environmental observations into sensory inputs.
		
		This method processes three types of inputs:
		1. Chemical signals from the environment
		2. Wall signals
		3. Internal signals
		
		Args:
			obs: Observation object containing chemicals, walls, and internal signals.
			neural-state: Object containing spatial positions (x), velocities (v),
						 sensor states (s), and masks (m).
			sensory_state: Current state of the sensory system.
			
		Returns:
			tuple: (I, sensory_state) where I is the combined sensory input and
				  sensory_state is the updated sensory state.
				  
		Raises:
			AssertionError: If neural-state is missing required attributes (x, v, s, m).
		"""

		C = obs.env
		inp = jnp.concatenate([obs.env, jnp.tile(obs.internal[:,None,None], (1, *C.shape[1:]))], axis=0)

		x, y = sensory_state.indices.T

		# Ic = jnp.where(sensory_state.on_border, jnp.sum(C[:,x,y].T * sensory_state.s[:,:C.shape[0]], axis=1), 0.0) # chemical input 

		# Ii = jnp.sum(sensory_state.s[:, C.shape[0]:] * obs.internal[None], axis=1) # internal input

		# I = Ic + Ii

		I = jnp.where(sensory_state.on_border, jnp.sum(inp[:,x,y].T * sensory_state.s, axis=1), 0.0)

		return I, sensory_state.energy_cost, sensory_state, {}









		