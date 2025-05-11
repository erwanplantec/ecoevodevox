from .base import SensoryInterface

import jax
from jax import numpy as jnp, random as jr, nn as jnn
import equinox as eqx
import equinox.nn as nn
from typing import Callable

class SpatiallyEmbeddedSensoryInterface(SensoryInterface):
	#-------------------------------------------------------------------
	sensor_expression_threshold: float=0.03
	sensor_activation: Callable=jax.nn.tanh
	border_threshold: float=0.9
	#-------------------------------------------------------------------
	def sensory_expression(self, policy_state):
		s = policy_state.s * policy_state.mask[:,None]
		s = jnp.where(jnp.abs(s)>self.sensor_expression_threshold, s, 0.0)
		s = self.sensor_activation(s) #neurons,ds
		return s
	#-------------------------------------------------------------------
	def encode(self, obs, policy_state, sensory_state):
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