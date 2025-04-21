from collections import namedtuple
from functools import partial
from typing import Callable
from flax.struct import PyTreeNode
import jax
import jax.numpy as jnp


type Action = jax.Array

class Interface(PyTreeNode):
	def encode(self, obs, net)->jax.Array:
		raise NotImplementedError
	def decode(self, net)->jax.Array:
		raise NotImplementedError


class SpatiallyEmbeddedNetworkInterface(Interface):

	sensor_expression_threshold: float=0.03
	sensor_activation: Callable=jax.nn.tanh
	motor_expression_threshold: float=0.03
	motor_activation: Callable=partial(jnp.clip, min=0.0, max=1.0)
	border_threshold: float=0.9
	max_neuron_force: float=0.1
	force_threshold_to_move: float=0.01
	max_motor_force: float=5.0

	def sensory_expression(self, net):
		s = net.s * net.mask[:,None]
		s = jnp.where(jnp.abs(s)>self.sensor_expression_threshold, s, 0.0)
		s = self.sensor_activation(s) #neurons,ds
		return s

	def motor_expression(self, net):
		m = net.m * net.mask[:,None]
		m = jnp.where(jnp.abs(m)>self.motor_expression_threshold, m, 0.0)
		m = self.motor_activation(m)
		on_border = jnp.any(jnp.abs(net.x)>self.border_threshold, axis=-1)
		m = jnp.where(on_border[:,None], m, 0.0); assert isinstance(m, jax.Array)
		return m

	def encode(self,obs, net)->jax.Array:
		xs = net.x
		s = self.sensory_expression(net)
		# ---
		C = obs.chemicals # mC,W,W
		W = obs.walls
		mC, D, _ = C.shape

		on_border = jnp.any(jnp.abs(xs)>self.border_threshold, axis=-1) #check if neuron is on border (epithelial layer)
		_xs = (xs.at[:,1].multiply(-1)+1)/2.0001 #make sure it does not reach upper bound
		coords = jnp.floor(_xs * D)
		coords = coords.astype(jnp.int16)
		j, i = coords.T

		Ic = jnp.where(on_border, jnp.sum(C[:,i,j].T * s[:,:mC], axis=1), 0.0) # chemical input #type:ignore
		Iw = jnp.where(on_border, jnp.sum(W[:,i,j].T * s[:,mC:mC+1], axis=1), 0.0) # walls input #type:ignore
		Ii = jnp.sum(s[:, mC+1:] * obs.internal, axis=1) # internal input #type:ignore

		I = Ic + Iw + Ii
		return I

	def decode(self, net)->Action:
		xs = net.x
		m = self.motor_expression(net)[:,0]
		v = net.v
		# ---
		xs = xs * jax.nn.one_hot(jnp.argmax(jnp.abs(xs), axis=-1), 2)

		on_top = xs[:,1] 	> self.border_threshold
		on_bottom = xs[:,1] < - self.border_threshold
		on_right = xs[:,0] 	> self.border_threshold
		on_left = xs[:,0] 	< - self.border_threshold

		forces = jnp.where(net.mask, jnp.clip(v*m, 0.0, self.max_neuron_force), 0.0) #forces applied by all neurons (N,)
		N_force = jnp.where(on_bottom, 	forces, 0.0).sum() 	#type:ignore
		S_force = jnp.where(on_top, 	forces, 0.0).sum()	#type:ignore
		E_force = jnp.where(on_left, 	forces, 0.0).sum() 	#type:ignore
		W_force = jnp.where(on_right, 	forces, 0.0).sum() 	#type:ignore

		directional_forces = jnp.array([N_force, S_force, E_force, W_force]) # 4,
		directions = jnp.array([[-1,0],[1,0],[0,1],[0,-1]], dtype=jnp.float16)
		
		net_directional_force = jnp.sum(directional_forces[:,None] * directions, axis=0) #2,
		move = jnp.where(jnp.abs(net_directional_force)>self.force_threshold_to_move, #if force on component is above threshold
						 jnp.clip(net_directional_force, -self.max_motor_force, self.max_motor_force), # move on unit
						 0.0).astype(jnp.float16) # don't move
		return move

