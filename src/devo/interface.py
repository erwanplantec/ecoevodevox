from .sensory import SensoryInterface
from .motor import MotorInterface
from .nn import NeuralModel, make_apply_init
from .core import *

from typing import Callable, Tuple
import jax, jax.random as jr, jax.numpy as jnp
from jaxtyping import Float16, Float, Int
import equinox as eqx
import math

type KeyArray = jax.Array

# ==================================================================

class AgentInterface(eqx.Module):
	# ------------------------------------------------------------------
	cfg: AgentConfig
	_neural_step: Callable[[NeuralParams,NeuralInput,NeuralState,KeyArray],tuple[NeuralState,Float]]
	_neural_init: Callable[[NeuralParams,KeyArray], NeuralState]
	_neural_fctry: Callable[[KeyArray], NeuralParams]
	_sensory_interface: SensoryInterface
	_motor_interface: MotorInterface
	_get_body_points: Callable[[Body], KeyArray]
	# ------------------------------------------------------------------
	def __init__(self,
	             cfg: AgentConfig,
				 sensory_interface: SensoryInterface,
				 motor_interface: MotorInterface,
				 neural_model_constructor: Callable[[KeyArray], NeuralModel]|None=None,
				 neural_step: Callable[[NeuralParams,NeuralInput,NeuralState,KeyArray],NeuralState]|None=None,
				 neural_init: Callable[[NeuralParams,KeyArray],NeuralState]|None=None,
				 neural_prms_fctry: Callable[[KeyArray], NeuralParams]|None=None):
		"""Initialize the AgentInterface. 
		The neural model can be provided either through a flax like interface by providing 
		neural_step, neural_init and neural_fctry or through an equinox like interface by providing the model constructor 
		as neural_model_constructor. If model provided with constructor, step, init and factory method will be created whcih additionally 
		take model parameters as first positional argument.
		
		Args:
		    cfg (AgentConfig): Description
		    sensory_interface (SensoryInterface): Interface for processing sensory inputs
		    motor_interface (MotorInterface): Interface for processing motor outputs
		    neural_model_constructor (Callable[[KeyArray], NeuralModel] | None, optional): Description
		    neural_step (Callable[[NeuralParams, NeuralInput, NeuralState, KeyArray], NeuralState] | None, optional): Description
		    neural_init (Callable[[NeuralParams, KeyArray], NeuralState] | None, optional): Function to initialize neural state from params and key
		    neural_prms_fctry (Callable[[KeyArray], NeuralParams] | None, optional): Description
		"""
		# ---
		if neural_model_constructor is not None:
			dummy_model = neural_model_constructor(jr.key(0))
			self._neural_step, self._neural_init = make_apply_init(dummy_model)
			self._neural_fctry = lambda key: eqx.filter(neural_model_constructor(key), eqx.is_array)
		else:
			assert (neural_step is not None) and (neural_init is not None) and (neural_prms_fctry is not None)
			self._neural_step = neural_step
			self._neural_init = neural_init
			self._neural_fctry = neural_prms_fctry
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
		self.cfg = self.cfg.replace(body_resolution=body_resolution)
		deltas = self.body_discretization_deltas()
		deltas_single_batch_dim = deltas.reshape(2,-1)

		@jax.jit
		def _get_body_points(body: Body):
			rotation_matrix = jnp.array([[jnp.cos(body.heading-jnp.pi/2), -jnp.sin(body.heading-jnp.pi/2)],
                             			 [jnp.sin(body.heading-jnp.pi/2), jnp.cos(body.heading-jnp.pi/2)]])
			rotated_deltas = jnp.matmul(rotation_matrix, deltas_single_batch_dim*body.size).reshape(2,*deltas.shape[1:])
			return body.pos[:,None,None]+rotated_deltas

		self._get_body_points = _get_body_points
	# ------------------------------------------------------------------
	def step(self, env_obs: jax.Array, state: AgentState, key: jax.Array)->Tuple[Action,AgentState,dict]:
		"""Make 1 update step of agent:
			encode -> neural update -> decode
		"""
		# 1. encode observation
		internals = jnp.stack(
	       [state.energy/self.cfg.max_energy,
	        state.age/self.cfg.max_age,
	        state.time_above_threshold/self.cfg.time_above_threshold_to_reproduce,
	        state.time_below_threshold/self.cfg.time_below_threshold_to_die], 
       	axis=-1)
		obs = Observation(env=env_obs, internal=internals)
		neural_input, sensory_energy_loss, sensory_state, sensory_info = self.encode_observation(obs, state.neural_state, state.sensory_state)
		# 2. neural update
		neural_state, neural_energy_loss = self.neural_step(state.genotype.neural_params, neural_input, state.neural_state, key)
		# 3. decode neural
		action, motor_energy_loss, motor_state, motor_info = self.decode_neural(neural_state, state.motor_state)
		# 4. compute energy loss (size, basal, motor, neural, sensory)
		size_energy_loss = self.cfg.size_energy_cost * state.genotype.body_size
		energy_loss = size_energy_loss + self.cfg.basal_energy_loss + motor_energy_loss + neural_energy_loss + sensory_energy_loss
		energy = state.energy - energy_loss

		state = state.replace(
			neural_state=neural_state, 
			motor_state=motor_state, 
			sensory_state=sensory_state, 
			energy=energy,
			age=state.age+1  
		)

		infos = {"motor_energy_loss": motor_energy_loss, 
				 "neural_energy_loss": neural_energy_loss, 
				 "sensory_energy_loss": sensory_energy_loss, 
				 **motor_info,
				 **sensory_info}

		return action, state, infos
	# ------------------------------------------------------------------
	def init(self, genotype: Genotype, position: jax.Array, heading: jax.Array, 
	         id_: UInt32, parent_id_: UInt32|None=None, generation: UInt32|None=None, 
	         *, key: jax.Array)->AgentState:
		"""Initialize the agent state (neural, sensory, motor, body size)"""
		# ---
		ks, kp, km = jr.split(key, 3)
		# --- 1. init neural (nn) state ---
		neural_state = self.neural_init(genotype.neural_params, kp)
		# --- 2. init sensory interface state ---
		sensory_state = self._sensory_interface.init(neural_state, ks)
		# --- 3. init motor int. state ---
		motor_state = self._motor_interface.init(neural_state, key=km)
		# --- 4. instantiate body ----
		body_size = jnp.clip(genotype.body_size, self.cfg.min_body_size, self.cfg.max_body_size)
		body = Body(pos=position, heading=heading, size=body_size)
		state = AgentState(genotype=genotype,
		                   body=body,
		                   motor_state=motor_state,
		                   sensory_state=sensory_state,
		                   neural_state=neural_state,
		                   alive=jnp.ones((), dtype=jnp.bool),
		                   age=jnp.ones((), dtype=jnp.uint16),
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
		                     time_above_threshold = jnp.where(has_reproduced, 0, state.time_above_threshold),
		                     n_offsprings = jnp.where(has_reproduced, state.n_offsprings+1, state.n_offsprings))
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
	def neural_step(self, neural_params: NeuralParams, neural_input: NeuralInput, 
		neural_state: NeuralState, key: jax.Array)->NeuralState:
		return self._neural_step(neural_params, neural_input, neural_state, key)
	# ------------------------------------------------------------------
	def neural_init(self, neural_params: NeuralParams, key: jax.Array)->NeuralState:
		return self._neural_init(neural_params, key)
	# ------------------------------------------------------------------
	def neural_fctry(self, key: jax.Array)->NeuralParams:
		return self._neural_fctry(key)
	# ------------------------------------------------------------------
	def move(self, action: Action, body: Body)->Body:
		return self._motor_interface.move(action,body)
	# ------------------------------------------------------------------
	def decode_neural(self, neural_state: NeuralState, motor_state: MotorState)->tuple[Action,Float16,MotorState,dict]:
		return self._motor_interface.decode(neural_state, motor_state)
	# ------------------------------------------------------------------
	def encode_observation(self, obs: Observation, neural_state: NeuralState, sensory_state: SensoryState)->tuple[NeuralInput,Float16,SensoryState,dict]:
		return self._sensory_interface.encode(obs, neural_state, sensory_state)
	# ------------------------------------------------------------------
	def get_body_points(self, body: Body) -> jax.Array:
		return self._get_body_points(body)
	# ------------------------------------------------------------------
	def body_discretization_deltas(self) -> jax.Array:
		body_resolution = self.cfg.body_resolution; assert body_resolution is not None
		return jnp.stack(
			[jnp.linspace(-0.5, 0.4999, body_resolution)[None,:].repeat(body_resolution, 0),
			 jnp.linspace(-0.5, 0.4999, body_resolution)[:,None].repeat(body_resolution, 1)]
		)
	# ------------------------------------------------------------------


#=======================================================================