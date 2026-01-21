from ..core import Observation, PolicyState
from .core import SensoryInterface

import jax
from jax import numpy as jnp, random as jr, nn as jnn
import equinox as eqx
import equinox.nn as nn
from typing import Callable
from flax.struct import PyTreeNode
from jaxtyping import Bool, Float, PyTree

type Observation=PyTree

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
	def sensory_expression(self, policy_state)->jax.Array:
		"""Process the sensory expression from the policy state.
		
		Args:
			policy_state: An object containing sensor states (s) and masks.
			
		Returns:
			jnp.ndarray: Processed sensory signals after thresholding and activation.
		"""
		s = policy_state.s * policy_state.mask[:,None]
		s = jnp.where(s>self.sensor_expression_threshold, s, 0.0); assert isinstance(s, jax.Array)
		return s
	#-------------------------------------------------------------------
	def init(self, policy_state, key):
		# ---
		assert hasattr(policy_state, "x") #make sure network is spatially embedded
		assert hasattr(policy_state, "v") #make sure neurons have activation
		assert hasattr(policy_state, "s") #make sure neurons have sensory expression
		# ---
		xs = policy_state.x
		s = self.sensory_expression(policy_state)
		on_border = jnp.any(jnp.abs(xs)>self.border_threshold, axis=-1) #check if neuron is on border (epithelial layer)
		_xs = (xs+1)/2.0001 #make sure it does not reach upper bound
		coords = jnp.floor(_xs * self.body_resolution)
		coords = coords.astype(jnp.int16)

		return State(on_border=on_border, 
			   		 s=s, 
			  	 	 mask=policy_state.mask, 
					 indices=coords, 	
					 energy_cost=jnp.astype(self.sensor_energy_cost*s.sum(), jnp.float16))
	#-------------------------------------------------------------------
	def encode(self, obs: Observation, policy_state: PolicyState, sensory_state: State):
		"""Encode environmental observations into sensory inputs.
		
		This method processes three types of inputs:
		1. Chemical signals from the environment
		2. Wall signals
		3. Internal signals
		
		Args:
			obs: Observation object containing chemicals, walls, and internal signals.
			policy_state: Object containing spatial positions (x), velocities (v),
						 sensor states (s), and masks (m).
			sensory_state: Current state of the sensory system.
			
		Returns:
			tuple: (I, sensory_state) where I is the combined sensory input and
				  sensory_state is the updated sensory state.
				  
		Raises:
			AssertionError: If policy_state is missing required attributes (x, v, s, m).
		"""

		C = obs.chemicals # mC, W, W
		W = obs.walls
		nC, D, _ = C.shape
		nW, *_ = W.shape
		nI, *_ = obs.internal.shape

		i, j = sensory_state.indices.T
		Ic = jnp.where(sensory_state.on_border, jnp.sum(C[:,i,j].T * sensory_state.s[:,:nC], axis=1), 0.0) # chemical input #type:ignore
		Iw = jnp.where(sensory_state.on_border, jnp.sum(W[:,i,j].T * sensory_state.s[:,nC:nC+nW], axis=1), 0.0) # walls input #type:ignore
		Ii = jnp.sum(sensory_state.s[:, nC+nW:nC+nW+nI] * obs.internal[None], axis=1) # internal input #type:ignore

		I = Ic + Iw + Ii

		return I, sensory_state.energy_cost, sensory_state, {}