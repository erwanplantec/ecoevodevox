from .sensory import SensoryInterface
from .motor import MotorInterface
from ..nn import Policy
from .core import *

from typing import Callable, Tuple
import jax, jax.random as jr, jax.numpy as jnp
from flax.struct import PyTreeNode
from jaxtyping import PyTree, Bool, Int16, UInt16, UInt32, Float16, Float
import equinox as eqx
from functools import partial


class AgentInterface(eqx.Module):
	#-------------------------------------------------------------------
	_policy_apply: Callable[[PolicyParams,PolicyInput,PolicyState,jax.Array],PolicyState]
	_policy_init: Callable[[PolicyParams,jax.Array], PolicyState]
	_policy_fctry: Callable[[jax.Array], PolicyParams]
	_sensory_interface: SensoryInterface
	_motor_interface: MotorInterface
	_full_body_pos: Callable
	basal_energy_loss: Float16
	#-------------------------------------------------------------------
	def __init__(self,
				 policy_apply: Callable[[PolicyParams,PolicyInput,PolicyState,jax.Array],PolicyState],
				 policy_init: Callable[[PolicyParams,jax.Array],PolicyState],
				 policy_fctry: Callable[[jax.Array], PolicyParams],
				 sensory_interface: SensoryInterface,
				 motor_interface: MotorInterface,
				 size: float=3.0,
				 body_resolution: int|None=None,
				 basal_energy_loss: float=0.0):
		# ---
		self._policy_apply = policy_apply
		self._policy_init = policy_init
		self._policy_fctry = policy_fctry
		self._sensory_interface = sensory_interface
		self._motor_interface = motor_interface
		# ---
		resolution = int(jnp.ceil(size+1)) if body_resolution is None else body_resolution
		deltas = jnp.stack(
			[jnp.linspace(-size/2, size/2.0001, resolution)[:,None].repeat(resolution, 1),
			 jnp.linspace(-size/2, size/2.0001, resolution)[None,:].repeat(resolution, 0)]
		)
		deltas_single_batch_dim = deltas.reshape(2,-1)

		@jax.jit
		def _get_body_points(pos: jax.Array, heading: Float):
			rotation_matrix = jnp.array([[jnp.cos(heading), -jnp.sin(heading)],
                             			 [jnp.sin(heading), jnp.cos(heading)]])
			rotated_deltas = jnp.matmul(rotation_matrix, deltas_single_batch_dim).reshape(2,*deltas.shape[1:])
			return pos[:,None,None]+rotated_deltas
		self._full_body_pos = _get_body_points
		self.basal_energy_loss = jnp.asarray(basal_energy_loss, dtype=jnp.float16)
	#-------------------------------------------------------------------
	def step(self, obs: Observation, state: AgentState, key: jax.Array)->Tuple[Action,AgentState,dict]:
		"""Make 1 update step of agent:
			encode -> policy update -> decode
		"""
		policy_input, sensory_state = self.encode_observation(obs, state.policy_state, state.sensory_state)
		policy_state, policy_energy_loss = self.policy_apply(state.policy_params, policy_input, state.policy_state, key)
		action, energy_loss, motor_state, motor_info = self.decode_policy(policy_state, state.motor_state)
		
		energy = state.energy - energy_loss - self.basal_energy_loss

		state = state.replace(
			policy_state=policy_state, 
			motor_state=motor_state, 
			sensory_state=sensory_state, 
			energy=energy,
			age=state.age+1  
		)

		infos = {"motor_energy_loss": energy_loss, "policy_energy_loss": policy_energy_loss, **motor_info}

		return action, state, infos
	#-------------------------------------------------------------------
	def init(self, policy_params: PolicyParams, key: jax.Array)->tuple[PolicyState,SensoryState,MotorState]:
		ks, kp, km = jr.split(key, 3)
		policy_state = self._policy_init(policy_params, kp)
		sensory_state = self._sensory_interface.init(policy_state, ks)
		motor_state = self._motor_interface.init(policy_state, km)
		return policy_state, sensory_state, motor_state
	#-------------------------------------------------------------------
	def policy_apply(self, policy_params: PolicyParams, policy_input: PolicyInput, 
		policy_state: PolicyState, key: jax.Array)->PolicyState:
		return self._policy_apply(policy_params, policy_input, policy_state, key)
	#-------------------------------------------------------------------
	def policy_init(self, policy_params: PolicyParams, key: jax.Array)->PolicyState:
		return self._policy_init(policy_params, key)
	#-------------------------------------------------------------------
	def policy_fctry(self, key: jax.Array)->PolicyParams:
		return self._policy_fctry(key)
	#-------------------------------------------------------------------
	def move(self, action: Action, position: Position):
		return self._motor_interface.move(action, position)
	#-------------------------------------------------------------------
	def decode_policy(self, policy_state: PolicyState, motor_state: MotorState):
		return self._motor_interface.decode(policy_state, motor_state)
	#-------------------------------------------------------------------	
	def encode_observation(self, obs: Observation, policy_state: PolicyState, sensory_state: SensoryState)->Action:
		return self._sensory_interface.encode(obs, policy_state, sensory_state)
	#-------------------------------------------------------------------
	def full_body_pos(self, position: Position)->jax.Array:
		return self._full_body_pos(position.pos, position.heading)
	#-------------------------------------------------------------------

#=======================================================================
#=======================================================================
#=======================================================================

class CiliasMotorInterfaceState(PyTreeNode):
	on_top: jax.Array
	on_bottom: jax.Array
	on_right: jax.Array
	on_left: jax.Array

class CiliasMotorInterface(MotorInterface):
	#-------------------------------------------------------------------
	motor_expression_threshold: float=0.03
	motor_activation: Callable=partial(jnp.clip, min=0.0, max=1.0)
	border_threshold: float=0.9
	max_neuron_force: float=0.1
	force_threshold_to_move: float=0.01
	max_motor_force: float=5.0
	energy_per_force_unit: float=0.0
	#-------------------------------------------------------------------
	def motor_expression(self, policy_state):
		m = policy_state.m * policy_state.mask[:,None]
		m = jnp.where(jnp.abs(m)>self.motor_expression_threshold, m, 0.0)
		m = self.motor_activation(m)
		on_border = jnp.any(jnp.abs(policy_state.x)>self.border_threshold, axis=-1)
		m = jnp.where(on_border[:,None], m, 0.0); assert isinstance(m, jax.Array)
		return m
	#-------------------------------------------------------------------
	def init(self, policy_state: PolicyState, key: jax.Array) -> MotorState:
		xs = policy_state.x
		on_top = xs[:,1] 	> self.border_threshold
		on_bottom = xs[:,1] < - self.border_threshold
		on_right = xs[:,0] 	> self.border_threshold
		on_left = xs[:,0] 	< - self.border_threshold

		return CiliasMotorInterfaceState(on_top, on_bottom, on_right, on_left)
	#-------------------------------------------------------------------
	def decode(self, policy_state: PolicyState, motor_state: CiliasMotorInterfaceState):
		m = self.motor_expression(policy_state)[:,0]
		v = policy_state.v
		# ---

		forces = jnp.where(policy_state.mask, jnp.clip(v*m, 0.0, self.max_neuron_force), 0.0) #forces applied by all neurons (N,)
		
		N_force = jnp.where(motor_state.on_bottom, 	forces, 0.0).sum() 	
		S_force = jnp.where(motor_state.on_top, 	forces, 0.0).sum()	
		E_force = jnp.where(motor_state.on_left, 	forces, 0.0).sum() 	
		W_force = jnp.where(motor_state.on_right, 	forces, 0.0).sum() 	

		directional_forces = jnp.array([N_force, S_force, E_force, W_force]) # 4,
		directions = jnp.array([[1,0],[-1,0],[0,1],[0,-1]], dtype=jnp.float16)
		
		net_directional_force = jnp.sum(directional_forces[:,None] * directions, axis=0) #2,
		move = jnp.where(jnp.abs(net_directional_force)>self.force_threshold_to_move, #if force on component is above threshold
						 jnp.clip(net_directional_force, -self.max_motor_force, self.max_motor_force), # move on unit
						 0.0).astype(jnp.float16) # don't move

		energy_loss = jnp.sum(forces * self.energy_per_force_unit)
		
		return move, energy_loss, motor_state

































