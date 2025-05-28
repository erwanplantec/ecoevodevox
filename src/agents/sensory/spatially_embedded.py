from .base import SensoryInterface

import jax
from jax import numpy as jnp, random as jr, nn as jnn
import equinox as eqx
import equinox.nn as nn
from typing import Callable

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
	sensor_expression_threshold: float=0.03
	sensor_activation: Callable=jax.nn.tanh
	border_threshold: float=0.9
	#-------------------------------------------------------------------
	def sensory_expression(self, policy_state):
		"""Process the sensory expression from the policy state.
		
		Args:
			policy_state: An object containing sensor states (s) and masks.
			
		Returns:
			jnp.ndarray: Processed sensory signals after thresholding and activation.
		"""
		s = policy_state.s * policy_state.mask[:,None]
		s = jnp.where(s>self.sensor_expression_threshold, s, 0.0)
		return s
	#-------------------------------------------------------------------
	def encode(self, obs, policy_state, sensory_state):
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
		# ---
		assert hasattr(policy_state, "x")
		assert hasattr(policy_state, "v")
		assert hasattr(policy_state, "s")
		assert hasattr(policy_state, "m")
		# ---
		xs = policy_state.x
		s = self.sensory_expression(policy_state)
		# ---
		C = obs.chemicals # mC,W,W
		W = obs.walls
		mC, D, _ = C.shape

		on_border = jnp.any(jnp.abs(xs)>self.border_threshold, axis=-1) #check if neuron is on border (epithelial layer)
		_xs = (xs+1)/2.0001 #make sure it does not reach upper bound
		coords = jnp.floor(_xs * D)
		coords = coords.astype(jnp.int16)
		i, j = coords.T

		Ic = jnp.where(on_border, jnp.sum(C[:,i,j].T * s[:,:mC], axis=1), 0.0) # chemical input #type:ignore
		Iw = jnp.where(on_border, jnp.sum(W[:,i,j].T * s[:,mC:mC+1], axis=1), 0.0) # walls input #type:ignore
		Ii = jnp.sum(s[:, mC+1:] * obs.internal, axis=1) # internal input #type:ignore

		I = Ic + Iw + Ii

		return I, sensory_state