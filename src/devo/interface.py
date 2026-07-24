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
		# `.replace` rather than rebuilding: this only casts the energy/size scalars to float16, so
		# every other field must carry over untouched. Reconstructing AgentConfig here instead
		# silently reset any field not listed back to its class default — which is how the
		# configured time_below_threshold_to_die / time_above_threshold_to_reproduce were being
		# dropped, and would drop each new field added to AgentConfig.
		self.cfg = cfg.replace(basal_energy_loss=jnp.asarray(cfg.basal_energy_loss, dtype=jnp.float16),
		                       size_energy_cost=jnp.asarray(cfg.size_energy_cost, dtype=jnp.float16),
		                       min_body_size=jnp.asarray(cfg.min_body_size, dtype=jnp.float16),
		                       max_body_size=jnp.asarray(cfg.max_body_size, dtype=jnp.float16),
		                       init_energy=jnp.asarray(cfg.init_energy, dtype=jnp.float16),
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
	def step(self, env_obs: jax.Array, state: AgentState, key: jax.Array)->Tuple[AgentState,dict]:
		"""Make 1 update step of agent:
			encode -> neural update -> actuate (moves the body). Returns (new_state, infos).
		"""
		# 1. encode observation
		internals = jnp.stack(
	       [state.energy/self.cfg.max_energy,
	        state.age/self.cfg.max_age,
	        state.time_above_threshold/self.cfg.time_above_threshold_to_reproduce,
	        state.time_below_threshold/self.cfg.time_below_threshold_to_die], 
       	axis=-1)
		key_neural, key_motor = jr.split(key)
		obs = Observation(env=env_obs, internal=internals)
		neural_input, sensory_energy_loss, sensory_state, sensory_info = self.encode_observation(obs, state.neural_state, state.sensory_state)
		# 2. neural update
		neural_state, neural_energy_loss = self.neural_step(state.genotype.neural_params, neural_input, state.neural_state, key_neural)
		# 3. actuate: turn the neural/motor state into the moved body + the action's energy cost.
		# The body is moved here (raw, unwrapped); the world wraps it toroidally afterwards. The
		# `action` itself is only exposed via motor_info["action"] for behaviour analysis.
		new_body, motor_energy_loss, motor_state, motor_info = self._motor_interface.actuate(
			neural_state, state.motor_state, state.body, key_motor)
		# distance from the true (pre-wrap) displacement; gated on alive so corpses don't accrue it
		step_dist = jnp.linalg.norm((new_body.pos - state.body.pos).astype(jnp.float32), axis=-1)
		distance = state.distance_travelled + jnp.where(state.alive, step_dist, 0.0)
		# lifetime turning: |heading change| wrapped to [-pi, pi] so crossing 0/2pi isn't counted as
		# a near-full turn (heading is stored mod 2pi). /age later gives the mean angular speed.
		dtheta = jnp.mod((new_body.heading - state.body.heading).astype(jnp.float32) + jnp.pi, 2*jnp.pi) - jnp.pi
		total_turn = state.total_abs_turn + jnp.where(state.alive, jnp.abs(dtheta), 0.0)
		# 4. compute energy loss (size, basal, motor, neural, sensory).
		# Maintenance (metabolic rate) scales as L**size_energy_exponent, default 1.5 = Kleiber (see
		# AgentConfig). Uses the realized (clipped) body size, not the raw genotype value which
		# mutation can drive negative.
		size_energy_loss = self.cfg.size_energy_cost * state.body.size ** self.cfg.size_energy_exponent
		energy_loss = size_energy_loss + self.cfg.basal_energy_loss + motor_energy_loss + neural_energy_loss + sensory_energy_loss
		energy = state.energy - energy_loss

		state = state.replace(
			neural_state=neural_state,
			motor_state=motor_state,
			sensory_state=sensory_state,
			body=new_body,
			energy=energy,
			age=state.age+1,
			distance_travelled=distance,
			total_abs_turn=total_turn,
		)

		# drop the raw action vector from the propagated infos (it is per-agent variable-length and
		# would break metric histogramming); its norm stays as a scalar
		motor_info = {k: v for k, v in motor_info.items() if k != "action"}
		infos = {"motor_energy_loss": motor_energy_loss,
				 "neural_energy_loss": neural_energy_loss,
				 "sensory_energy_loss": sensory_energy_loss,
				 **motor_info,
				 **sensory_info}

		return state, infos
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
		# clip the genotype's body size into the valid range and write it back, so the stored /
		# inherited genotype stays in range every generation. Otherwise mutation drives it below
		# min (even negative), and a negative size makes the size energy cost negative -> free
		# energy. Cast so body.size stays float16 (clip promotes to float32).
		body_size = jnp.clip(genotype.body_size, self.cfg.min_body_size, self.cfg.max_body_size).astype(self.cfg.min_body_size.dtype)
		genotype = genotype.replace(body_size=body_size)
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
						   distance_travelled=jnp.zeros((), dtype=jnp.float32),
						   total_abs_turn=jnp.zeros((), dtype=jnp.float32),
						   generation=generation if generation is not None else jnp.ones((), dtype=jnp.uint32),
						   id_=id_,
						   parent_id_=parent_id_ if parent_id_ is not None else jnp.zeros((), dtype=jnp.uint32))
		return state
	# ------------------------------------------------------------------
	def update_energy(self, state: AgentState, energy_intake: Float16)->AgentState:
		return state.replace(energy=jnp.clip(state.energy + energy_intake, -jnp.inf, self.cfg.max_energy))
	# ------------------------------------------------------------------
	def update_after_reproduction(self, state: AgentState, has_reproduced: Bool) -> AgentState:
		# reproduction cost scales as L**reproduction_energy_exponent (default 2.0 ~ body mass:
		# building a bigger body is proportionally more expensive); the child inherits the parent's
		# size, so use it here
		repro_cost = has_reproduced * self.cfg.reproduction_energy_cost * state.body.size ** self.cfg.reproduction_energy_exponent
		return state.replace(energy = state.energy - repro_cost,
		                     time_above_threshold = jnp.where(has_reproduced, 0, state.time_above_threshold),
		                     n_offsprings = jnp.where(has_reproduced, state.n_offsprings+1, state.n_offsprings))
	# ------------------------------------------------------------------
	def is_eating(self, state: AgentState)->Bool:
		# satiation threshold, as a fraction of max_energy. Gating on `energy < max_energy` alone
		# is vacuous: per-step costs drop an agent below max immediately, so it re-tops-up every
		# step and food is grazed flat as fast as it grows.
		return state.alive & (state.energy < self.cfg.eat_energy_fraction * self.cfg.max_energy)
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
	def actuate(self, neural_state: NeuralState, motor_state: MotorState, body: Body, key: jax.Array)->tuple[Body,Float16,MotorState,dict]:
		return self._motor_interface.actuate(neural_state, motor_state, body, key)
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