from .sensory import SensoryInterface
from .motor import MotorInterface
from .core import *

from typing import Callable, Tuple
import jax, jax.random as jr, jax.numpy as jnp
from jaxtyping import Float16, Float, Int
import equinox as eqx
import math

# ==================================================================

class AgentInterface(eqx.Module):
	# ------------------------------------------------------------------
	cfg: AgentConfig
	_policy_apply: Callable[[PolicyParams,PolicyInput,PolicyState,jax.Array],PolicyState]
	_policy_init: Callable[[PolicyParams,jax.Array], PolicyState]
	_policy_fctry: Callable[[jax.Array], PolicyParams]
	_sensory_interface: SensoryInterface
	_motor_interface: MotorInterface
	_get_body_points: Callable[[Body], jax.Array]
	# ------------------------------------------------------------------
	def __init__(self,
	             cfg: AgentConfig,
				 policy_apply: Callable[[PolicyParams,PolicyInput,PolicyState,jax.Array],PolicyState],
				 policy_init: Callable[[PolicyParams,jax.Array],PolicyState],
				 policy_fctry: Callable[[jax.Array], PolicyParams],
				 sensory_interface: SensoryInterface,
				 motor_interface: MotorInterface):
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
		self.cfg = AgentConfig(basal_energy_loss=jnp.asarray(cfg.basal_energy_loss, dtype=jnp.float16),
		                       size_energy_cost=jnp.asarray(cfg.size_energy_cost, dtype=jnp.float16),
		                       min_body_size=jnp.asarray(cfg.min_body_size, dtype=jnp.float16),
		                       max_body_size=jnp.asarray(cfg.max_body_size, dtype=jnp.float16),
		                       body_resolution=cfg.body_resolution,
		                       init_energy=jnp.asarray(cfg.init_energy, dtype=jnp.float16),
		                       max_age=cfg.max_age,
		                       max_energy=jnp.asarray(cfg.max_energy, dtype=jnp.float16),
		                       reproduction_energy_cost=jnp.asarray(cfg.reproduction_energy_cost, dtype=jnp.float16))
		# ---
		body_resolution = cfg.body_resolution if cfg.body_resolution is not None else math.ceil(int(cfg.max_body_size)) + 1
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
	# ------------------------------------------------------------------
	def step(self, env_obs: jax.Array, state: AgentState, key: jax.Array)->Tuple[Action,AgentState,dict]:
		"""Make 1 update step of agent:
			encode -> policy update -> decode
		"""
		# 1. encode observation
		internals = jnp.stack(
	       [state.energy/self.cfg.max_energy,
	        state.age/self.cfg.max_age,
	        state.time_above_threshold/self.cfg.time_above_threshold_to_reproduce,
	        state.time_below_threshold/self.cfg.time_below_threshold_to_die], 
       	axis=-1)
		obs = Observation(env=env_obs, internal=internals)
		policy_input, sensory_energy_loss, sensory_state, sensory_info = self.encode_observation(obs, state.policy_state, state.sensory_state)
		# 2. policy update
		policy_state, policy_energy_loss = self.policy_apply(state.genotype.policy_params, policy_input, state.policy_state, key)
		# 3. decode policy
		action, motor_energy_loss, motor_state, motor_info = self.decode_policy(policy_state, state.motor_state)
		# 4. compute energy loss (size, basal, motor, policy, sensory)
		size_energy_loss = self.cfg.size_energy_cost * state.genotype.body_size
		energy_loss = size_energy_loss + self.cfg.basal_energy_loss + motor_energy_loss + policy_energy_loss + sensory_energy_loss
		energy = state.energy - energy_loss

		state = state.replace(
			policy_state=policy_state, 
			motor_state=motor_state, 
			sensory_state=sensory_state, 
			energy=energy,
			age=state.age+1  
		)

		infos = {"motor_energy_loss": motor_energy_loss, 
				 "policy_energy_loss": policy_energy_loss, 
				 "sensory_energy_loss": sensory_energy_loss, 
				 **motor_info,
				 **sensory_info}

		return action, state, infos
	# ------------------------------------------------------------------
	def init(self, genotype: Genotype, position: jax.Array, heading: jax.Array, 
	         id_: UInt32, parent_id_: UInt32|None=None, generation: UInt32|None=None, 
	         *, key: jax.Array)->AgentState:
		"""Initialize the agent state (policy, sensory, motor, body size)"""
		# ---
		ks, kp, km = jr.split(key, 3)
		# --- 1. init policy (nn) state ---
		policy_state = self._policy_init(genotype.policy_params, kp)
		# --- 2. init sensory interface state ---
		sensory_state = self._sensory_interface.init(policy_state, ks)
		# --- 3. init motor int. state ---
		motor_state = self._motor_interface.init(policy_state, km)
		# --- 4. instantiate body ----
		body_size = jnp.clip(genotype.body_size, self.cfg.min_body_size, self.cfg.max_body_size)
		body = Body(pos=position, heading=heading, size=body_size)
		state = AgentState(genotype=genotype,
		                   body=body,
		                   motor_state=motor_state,
		                   sensory_state=sensory_state,
		                   policy_state=policy_state,
		                   alive=jnp.ones((), dtype=jnp.bool),
		                   age=jnp.zeros((), dtype=jnp.uint16),
		                   energy=self.cfg.init_energy,
		                   time_above_threshold=jnp.zeros((), dtype=jnp.uint16),
		                   time_below_threshold=jnp.zeros((), dtype=jnp.uint16),
						   n_offsprings=jnp.zeros((), jnp.uint16),
						   generation=generation if generation is not None else jnp.ones((), dtype=jnp.uint32),
						   id_=id_,
						   parent_id_=parent_id_ if parent_id_ is not None else jnp.zeros((), dtype=jnp.uint32))
		return state
	# ------------------------------------------------------------------
	def update_energy(self, state: AgentState, energy_intake: Float16)->AgentState:
		return state.replace(energy=jnp.clip(state.energy + energy_intake, -jnp.inf, self.cfg.max_energy))
	# ------------------------------------------------------------------
	def update_after_reproduction(self, state: AgentState, has_reproduced: Bool) -> AgentState:
		return state.replace(energy = state.energy - (has_reproduced * self.cfg.reproduction_energy_cost),
		                     time_above_threshold = jnp.where(has_reproduced, 0, state.time_above_threshold))
	# ------------------------------------------------------------------
	def is_eating(self, state: AgentState)->Bool:
		return state.alive & (state.energy < self.cfg.max_energy)
	# ------------------------------------------------------------------
	def is_reproducing(self, state: AgentState)->Bool:
		return (state.time_above_threshold > self.cfg.time_above_threshold_to_reproduce) & state.alive
	# ------------------------------------------------------------------
	def is_dying(self, state: AgentState)->Bool:
		return (state.age > self.cfg.max_age) | (state.time_below_threshold > self.cfg.time_below_threshold_to_die) 
	# ------------------------------------------------------------------
	def policy_apply(self, policy_params: PolicyParams, policy_input: PolicyInput, 
		policy_state: PolicyState, key: jax.Array)->PolicyState:
		return self._policy_apply(policy_params, policy_input, policy_state, key)
	# ------------------------------------------------------------------
	def policy_init(self, policy_params: PolicyParams, key: jax.Array)->PolicyState:
		return self._policy_init(policy_params, key)
	# ------------------------------------------------------------------
	def policy_fctry(self, key: jax.Array)->PolicyParams:
		return self._policy_fctry(key)
	# ------------------------------------------------------------------
	def move(self, action: Action, body: Body)->Body:
		return self._motor_interface.move(action,body)
	# ------------------------------------------------------------------
	def decode_policy(self, policy_state: PolicyState, motor_state: MotorState)->tuple[Action,Float16,MotorState,dict]:
		return self._motor_interface.decode(policy_state, motor_state)
	# ------------------------------------------------------------------
	def encode_observation(self, obs: Observation, policy_state: PolicyState, sensory_state: SensoryState)->tuple[PolicyInput,Float16,SensoryState,dict]:
		return self._sensory_interface.encode(obs, policy_state, sensory_state)
	# ------------------------------------------------------------------
	def get_body_points(self, body: Body) -> jax.Array:
		return self._get_body_points(body)
	# ------------------------------------------------------------------

#=======================================================================