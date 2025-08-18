from .sensory import SensoryInterface
from .motor import MotorInterface
from .core import *

from typing import Callable, Tuple
import jax, jax.random as jr, jax.numpy as jnp
from jaxtyping import Float16, Float
import equinox as eqx

class AgentInterface(eqx.Module):
	#-------------------------------------------------------------------
	_policy_apply: Callable[[PolicyParams,PolicyInput,PolicyState,jax.Array],PolicyState]
	_policy_init: Callable[[PolicyParams,jax.Array], PolicyState]
	_policy_fctry: Callable[[jax.Array], PolicyParams]
	_sensory_interface: SensoryInterface
	_motor_interface: MotorInterface
	_get_body_points: Callable
	basal_energy_loss: Float16
	size_energy_cost: Float16
	min_body_size: Float
	max_body_size: Float
	#-------------------------------------------------------------------
	def __init__(self,
				 policy_apply: Callable[[PolicyParams,PolicyInput,PolicyState,jax.Array],PolicyState],
				 policy_init: Callable[[PolicyParams,jax.Array],PolicyState],
				 policy_fctry: Callable[[jax.Array], PolicyParams],
				 sensory_interface: SensoryInterface,
				 motor_interface: MotorInterface,
				 body_resolution: int=4,
				 basal_energy_loss: float=0.0,
				 size_energy_cost: float=0.0,
				 min_body_size: float=1.0,
				 max_body_size: float=10.0):
		"""Initialize the AgentInterface.
		
		Args:
			policy_apply: Function to apply policy given params, input, state, and key
			policy_init: Function to initialize policy state from params and key
			policy_fctry: Function to create policy params from key
			sensory_interface: Interface for processing sensory inputs
			motor_interface: Interface for processing motor outputs
			body_resolution: Resolution for body point discretization (default: 4)
			basal_energy_loss: Base energy loss per step (default: 0.0)
			min_body_size: Minimum allowed body size (default: 1.0)
			max_body_size: Maximum allowed body size (default: 10.0)
		"""
		# ---
		self._policy_apply = policy_apply
		self._policy_init = policy_init
		self._policy_fctry = policy_fctry
		self._sensory_interface = sensory_interface
		self._motor_interface = motor_interface
		self.basal_energy_loss = jnp.asarray(basal_energy_loss, dtype=jnp.float16)
		self.size_energy_cost = jnp.asarray(size_energy_cost, dtype=jnp.float16)
		self.min_body_size = jnp.asarray(min_body_size, dtype=jnp.float16)
		self.max_body_size = jnp.asarray(max_body_size, dtype=jnp.float16)
		# ---
		deltas = jnp.stack(
			[jnp.linspace(-0.5, 0.4999, body_resolution)[:,None].repeat(body_resolution, 1),
			 -jnp.linspace(-0.5, 0.4999, body_resolution)[None,:].repeat(body_resolution, 0)]
		).transpose(0,2,1)
		deltas_single_batch_dim = deltas.reshape(2,-1)

		@jax.jit
		def _get_body_points(body: Body):
			rotation_matrix = jnp.array([[jnp.cos(body.heading), -jnp.sin(body.heading)],
                             			 [jnp.sin(body.heading), jnp.cos(body.heading)]])
			rotated_deltas = jnp.matmul(rotation_matrix, deltas_single_batch_dim*body.size).reshape(2,*deltas.shape[1:])
			return body.pos[:,None,None]+rotated_deltas

		self._get_body_points = _get_body_points
	#-------------------------------------------------------------------
	def step(self, obs: Observation, state: AgentState, key: jax.Array)->Tuple[Action,AgentState,dict]:
		"""Make 1 update step of agent:
			encode -> policy update -> decode
		"""
		# 1. encode observation
		policy_input, sensory_energy_loss, sensory_state, sensory_info = self.encode_observation(obs, state.sensory_state)
		# 2. policy update
		policy_state, policy_energy_loss = self.policy_apply(state.genotype.policy_params, policy_input, state.policy_state, key)
		# 3. decode policy
		action, motor_energy_loss, motor_state, motor_info = self.decode_policy(policy_state, state.motor_state)
		# 4. compute energy loss (size, basal, motor, policy, sensory)
		size_energy_loss = self.size_energy_cost * state.genotype.body_size
		energy = state.energy - sensory_energy_loss - policy_energy_loss - motor_energy_loss - self.basal_energy_loss - size_energy_loss

		state = state.replace(
			policy_state=policy_state, 
			motor_state=motor_state, 
			sensory_state=sensory_state, 
			energy=energy,
			age=state.age+1  
		)

		infos = {"motor_energy_loss": motor_energy_loss, "policy_energy_loss": policy_energy_loss, "sensory_energy_loss": sensory_energy_loss, **motor_info}

		return action, state, infos
	#-------------------------------------------------------------------
	def init(self, genotype: Genotype, key: jax.Array)->tuple[PolicyState,SensoryState,MotorState,Float16]:
		"""Initialize the agent state (policy, sensory, motor, body size)"""
		# ---
		ks, kp, km = jr.split(key, 3)
		policy_state = self._policy_init(genotype.policy_params, kp)
		sensory_state = self._sensory_interface.init(policy_state, ks)
		motor_state = self._motor_interface.init(policy_state, km)
		body_size = jnp.clip(genotype.body_size, self.min_body_size, self.max_body_size)
		return policy_state, sensory_state, motor_state, body_size
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
	def move(self, action: Action, body: Body)->Body:
		return self._motor_interface.move(action,body)
	#-------------------------------------------------------------------
	def decode_policy(self, policy_state: PolicyState, motor_state: MotorState)->tuple[Action,Float16,MotorState,dict]:
		return self._motor_interface.decode(policy_state, motor_state)
	#-------------------------------------------------------------------	
	def encode_observation(self, obs: Observation, sensory_state: SensoryState)->tuple[PolicyInput,Float16,SensoryState,dict]:
		return self._sensory_interface.encode(obs, sensory_state)
	#-------------------------------------------------------------------
	def get_body_points(self, body: Body)->jax.Array:
		return self._get_body_points(body)
	#-------------------------------------------------------------------

#=======================================================================
#=======================================================================
#=======================================================================


































