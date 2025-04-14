from functools import partial
import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import jax.scipy as jsp
import equinox as eqx
import numpy as np
import equinox.nn as nn
from celluloid import Camera

import matplotlib.pyplot as plt

from typing import Callable, NamedTuple, Tuple, TypeVar
from jaxtyping import (
	Float, PyTree, Array,
	Bool,
	Int16, Int32,
	UInt8, UInt16, UInt32,
	Float16, Float32
)

# ======================== UTILS =============================

from .utils import *

type FoodMap = Bool[Array, "F H W"]
type KeyArray = jax.Array
type AgentState = PyTree
type AgentParams = jax.Array
type Action = jax.Array

def make_growth_convolution(env_size: tuple[int,int],
							reproduction_rates: jax.Array,
							dmins: jax.Array,
							dmaxs: jax.Array,
							inhib: float=-1.0,
							dtype: type=jnp.float32):
	"""Creates convolution function for food growth probabilities"""
	# ---
	H, W = env_size
	# ---
	assert (not H%2) and (not W%2)
	# ---
	mH, mW = H//2, W//2
	L = jnp.mgrid[-mH:mH,-mW:mW]
	D = jnp.linalg.norm(L, axis=0, keepdims=True)

	growth_kernels = ((D>=dmins[:,None,None]) & (D<=dmaxs[:,None,None])).astype(jnp.float32)
	growth_kernels = growth_kernels / growth_kernels.sum(axis=(1,2), keepdims=True)
	growth_kernels = growth_kernels * reproduction_rates[:,None,None]
	growth_kernels = jnp.where(D<dmins[:,None,None], inhib, growth_kernels); assert isinstance(growth_kernels,jax.Array)
	growth_kernels_fft = jnp.fft.fft2(jnp.fft.fftshift(growth_kernels, axes=(1,2))).astype(dtype)

	@jax.jit
	def _conv(F: Bool[Array, "F H W"])->jax.Array:
		F_fft = jnp.fft.fft2(F.astype(dtype))
		P = jnp.real(jnp.fft.ifft2(F_fft*growth_kernels_fft))
		P = jnp.where((P<0)|jnp.isclose(P,0.0), 0.0, P); assert isinstance(P, jax.Array)
		return P

	return _conv

def make_chemical_diffusion_convolution(env_size: tuple[int,int],
										diffusion_rates: jax.Array):
	# ---
	H, W = env_size
	# ---
	assert (not H%2) and (not W%2)
	# ---
	mH, mW = H//2, W//2
	L = jnp.mgrid[-mH:mH,-mW:mW]
	D = jnp.linalg.norm(L, axis=0, keepdims=True) #1,H,W

	diffusion_kernels = jnp.exp(-D/diffusion_rates[:,None,None]); assert isinstance(diffusion_kernels, jax.Array)
	diffusion_kernels_fft = jnp.fft.fft2(jnp.fft.fftshift(diffusion_kernels, axes=(1,2)))

	@jax.jit
	def _conv(C: Float[jax.Array, "C H W"])->Float[jax.Array, "C H W"]:
		C_fft = jnp.fft.fft2(C)
		res = jnp.real(jnp.fft.ifft2(C_fft * diffusion_kernels_fft))
		return res

	return _conv


# ============================================================

class Agent(NamedTuple):
	# --- 
	prms: jax.Array
	policy_state: PyTree
	# ---
	alive: Bool
	age: Int16
	position: Int16
	energy: Float16
	time_above_threshold: Int16
	time_below_threshold: Int16
	# ---
	reward: Float16
	reproduce: Bool
	# --- infos
	n_offsprings: UInt16
	generation: UInt32
	id_: UInt32
	parent_id_: UInt32
	# ---

class ChemicalType(NamedTuple):
	diffusion_rate: Float16

class FoodType(NamedTuple):
	growth_rate: Float32
	dmin: Float32
	dmax: Float32
	chemical_signature: Float32
	energy_concentration: Float32
	spontaneous_grow_prob: Float32
	initial_density: Float32

class EnvState(NamedTuple):
	agents: Agent
	food: FoodMap
	time: Int32
	last_agent_id: UInt32=0

class Observation(NamedTuple):
	chemicals: jax.Array
	internal: jax.Array
	walls: jax.Array

class GridWorld:
	# ---
	def __init__(
		self, 
		size: Tuple[int, int],
		# ---
		agent_fctry: Callable[[KeyArray], AgentParams], 
		agent_init: Callable[[AgentParams, KeyArray], AgentState],
		agent_apply: Callable[[AgentParams, Observation, AgentState, KeyArray], Tuple[Action,AgentState]],
		mutation_fn: Callable[[AgentParams,KeyArray],AgentParams], 
		# ---
		chemical_types: ChemicalType,
		food_types: FoodType,  
		# ---
		max_agents: int=1_024, 
		init_agents: int=256,
		passive_eating: bool=True,
		passive_reproduction: bool=True,
		predation: bool=False,
		max_age: int=1_000,
		field_of_view: int=1,
		birth_pool_size: int|None=None,
		# ---
		walls_density: float=0.0,
		# ---
		energy_reproduction_threshold: float=0.,
		reproduction_energy_cost: float=0.5,
		predation_energy_gain: float=5.,
		predation_energy_cost: float=0.1,
		base_energy_loss: float=0.05,
		max_energy: float=10.0,
		state_energy_cost_fn: Callable[[Agent],Float16]=lambda agent: jnp.zeros((), dtype=jnp.float16),
		# ---
		time_above_threshold_to_reproduce: int=20,
		time_below_threshold_to_die: int=10,
		initial_agent_energy: float=1.0,
		chemical_detection_threshold: float=0.01,
		deadly_walls: bool=True,
		size_apply_minibatches: int|None=None,
		size_init_minibatches: int|None=None,
		*,
		key: jax.Array):
		
		self.size = size
		self.walls = jr.bernoulli(key, walls_density, self.size)
		self.deadly_walls = deadly_walls

		self.food_types = jax.tree.map(lambda x: x.astype(jnp.float16), food_types)
		self.nb_food_types = food_types.growth_rate.shape[0]
		self.growth_conv = make_growth_convolution(self.size, 
												   food_types.growth_rate,
												   food_types.dmin, 
												   food_types.dmax)
	
		self.chemical_types = chemical_types
		self.chemicals_diffusion_conv = make_chemical_diffusion_convolution(self.size,
																			chemical_types.diffusion_rate)

		deltas = jnp.mgrid[-field_of_view:field_of_view+1, -field_of_view:field_of_view+1]
		@jax.jit
		def _vision_fn(x: jax.Array, pos: jax.Array):
			indices = jnp.mod(pos[:,None,None]+deltas, jnp.array(self.size, dtype=i16)[:,None,None])
			return x[:, *indices]
		self.vision_fn = _vision_fn

		self.mutation_fn = mutation_fn
		self.agent_fctry = agent_fctry
		self.agent_init = agent_init
		self.agent_apply = agent_apply
		
		dummy_agent_prms = agent_fctry(jr.key(1))
		dummy_policy_state = agent_init(dummy_agent_prms, jr.key(1))
		self.mapped_agent_fctry = jax.vmap(agent_fctry)
		if size_init_minibatches is None:
			self.mapped_agent_init = jax.vmap(lambda prms, key, msk: agent_init(prms, key)) #adds extra input for mask (useless in case of vmap)
		else:
			self.mapped_agent_init = minivmap(
				lambda prms, key, msk: agent_init(prms, key), 
				minibatch_size=size_init_minibatches, 
				check_func=lambda prms, key, msk: jnp.any(msk),
				default_output=dummy_policy_state
			) 
		if size_apply_minibatches is None:
			self.mapped_agent_apply = jax.vmap(lambda prms, obs, state, key, msk: agent_apply(prms, obs, state, key))
		else:
			dummy_output = (5, dummy_policy_state)
			self.mapped_agent_apply = minivmap(
				lambda prms, obs, state, key, msk: agent_apply(prms, obs, state, key),
				minibatch_size=size_apply_minibatches,
				check_func=lambda prms, obs, state, key, msk: jnp.any(msk),
				default_output=dummy_output)

		self.max_agents = max_agents
		self.init_agents = init_agents
		self.birth_pool_size = birth_pool_size if birth_pool_size is not None else max_agents
		self.predation = predation
		self.field_of_view = field_of_view
		self.passive_eating = passive_eating
		self.passive_reproduction = passive_reproduction
		self.energy_reproduction_threshold = energy_reproduction_threshold
		self.time_above_threshold_to_reproduce = time_above_threshold_to_reproduce
		self.time_below_threshold_to_die = time_below_threshold_to_die
		self.initial_agent_energy = initial_agent_energy
		self.reproduction_energy_cost = reproduction_energy_cost
		self.base_energy_loss = base_energy_loss
		self.max_energy = max_energy
		self.agent_scent_diffusion_kernel = jnp.array([[ 0.1 , 0.1 , 0.1 ],
													   [ 0.1 , 1.0 , 0.1 ],
													   [ 0.1 , 0.1 , 0.1 ]], dtype=f16)
		self.chemical_detection_threshold = chemical_detection_threshold
		self.predation_energy_gain = predation_energy_gain
		self.predation_energy_cost = predation_energy_cost
		self.state_energy_cost_fn = state_energy_cost_fn
		self.max_age = max_age

	# ---

	@property
	def n_actions(self):
		return 5 + int(not self.passive_eating) + int(not self.passive_reproduction) + int(self.predation)

	# ---

	def step(self, state: EnvState, key: jax.Array)->Tuple[EnvState,PyTree]:

		# --- 1. Update food sources ---
		key, key_food = jr.split(key)
		state = self._update_food(state, key_food)

		# --- 2. Get and apply actions ---
		key, key_action = jr.split(key)
		observations = self._get_observations(state)
		
		actions, policy_states = self.mapped_agent_apply(state.agents.prms, 
											 			 observations, 
											 			 state.agents.policy_state, 
											 			 jr.split(key_action, self.max_agents), 
											 			 state.agents.alive)
		actions = actions.astype(jnp.int16)
		state = eqx.tree_at(lambda s: s.agents.policy_state, state, policy_states)
		state, actions_data = self._apply_actions(state, actions)

		# --- 3. Die / reproduce ---
		state, update_agents_data = self._update_agents(state, key)

		state = state._replace(time=state.time+1)

		return (
			state, 
			dict(state=state, 
				 actions=actions, 
				 observations=observations,
				 **actions_data, 
				 **update_agents_data)
		)

	# ---

	def reset(self, key: jax.Array)->EnvState:
		key_food, key_agent = jr.split(key, 2)
		food_sources = self._init_food(key_food)
		agents = self._init_agents(key_agent)
		return EnvState(agents=agents, food=food_sources, time=0, last_agent_id=agents.id_.max())

	# ===================== AGENTs ========================

	def _init_agents(self, key):

		key_prms, key_pos, key_init, key_age = jr.split(key, 4)
		alive = jnp.arange(self.max_agents) < self.init_agents
		prms = jax.vmap(self.agent_fctry)(jr.split(key_prms,self.max_agents))
		policy_states = self.mapped_agent_init(prms, jr.split(key_init, alive.shape[0]), alive)
		positions = jr.randint(key_pos, (self.max_agents, 2), minval=1, maxval=jnp.array(self.size, dtype=i16)-1, dtype=i16)

		return Agent(
			# ---
			prms 				 = prms, 
			policy_state 		 = policy_states,
			# ---
			alive 				 = alive, 
			energy 				 = jnp.full((self.max_agents), self.initial_agent_energy, dtype=f16)*alive, 
			time_above_threshold = jnp.full((self.max_agents,), 0, dtype=ui16), 
			time_below_threshold = jnp.full((self.max_agents,), 0, dtype=ui16),
			position 			 = positions, 
			# ---
			reproduce 			 = jnp.full((self.max_agents,), False, dtype=bool),
			reward 				 = jnp.zeros((self.max_agents,), dtype=f16), 
			# ---
			age 				 = jnp.ones((self.max_agents), dtype=ui16), 
			n_offsprings 		 = jnp.zeros(self.max_agents, dtype=ui16),
			id_ 				 = jnp.where(alive, jnp.cumsum(alive, dtype=ui32), 0),
			parent_id_ 			 = jnp.zeros(self.max_agents, dtype=ui32),
			generation 			 = jnp.zeros(self.max_agents, dtype=ui16)
		)

	# ---

	def _update_agents(self, state: EnvState, key: jax.Array)->Tuple[EnvState, PyTree]:
		
		key_repr, key_mut, key_init = jr.split(key, 3)
		agents = state.agents

		# --- Update Energy ---
		state_dependant_energy_loss = jax.vmap(self.state_energy_cost_fn)(agents)
		energy_loss = self.base_energy_loss + state_dependant_energy_loss
		agents_energy = jnp.where(agents.alive, agents.energy-energy_loss, 0.0)
		agents = agents._replace(energy=agents_energy)
		
		below_threshold = agents.energy < 0.0
		above_threshold = agents.energy > self.energy_reproduction_threshold
		
		agents_tat = jnp.where(above_threshold&agents.alive, agents.time_above_threshold+1, 0); assert isinstance(agents_tat, jax.Array)
		agents_tbt = jnp.where(below_threshold&agents.alive, agents.time_below_threshold+1, 0); assert isinstance(agents_tbt, jax.Array)

		# --- 1. Death ---

		dead = (agents_tbt > self.time_below_threshold_to_die) | (agents.age > self.max_age)
		dead = dead & agents.alive
		avg_dead_age = jnp.where(dead, agents.age, 0).sum() / dead.sum() #type:ignore
		agents_alive = agents.alive & ( ~dead )
		agents_age = jnp.where(agents_alive, agents.age, 0)
		agents = agents._replace(alive=agents_alive, time_above_threshold=agents_tat, time_below_threshold=agents_tbt, age=agents_age)

		# --- 2. Reproduce ---

		# ---

		def _reproduce(reproducing: jax.Array, agents: Agent)->Agent:
			"""
			"""
			free_buffer_spots = ~agents.alive # N,
			_, parents_buffer_id = jax.lax.top_k(reproducing+jr.uniform(key_repr,reproducing.shape,minval=-0.1,maxval=0.1), self.birth_pool_size)
			parents_mask = reproducing[parents_buffer_id]
			parents_prms = agents.prms[parents_buffer_id]
			is_free, childs_buffer_id = jax.lax.top_k(free_buffer_spots, self.birth_pool_size)
			childs_mask = parents_mask & is_free #is a child if parent was actually reproducing and there are free buffer spots
			childs_buffer_id = jnp.where(childs_mask, childs_buffer_id, self.max_agents) # assign wrong index if not born
			parents_buffer_id = jnp.where(childs_mask, parents_buffer_id, self.max_agents)

			childs_alive = childs_mask
			childs_prms = jax.vmap(self.mutation_fn)(parents_prms, jr.split(key_mut, self.birth_pool_size))
			childs_policy_states = self.mapped_agent_init(childs_prms, jr.split(key_init, self.birth_pool_size), childs_mask)
			childs_energy = jnp.full(self.birth_pool_size, self.initial_agent_energy, dtype=f16)
			childs_positions = agents.position[parents_buffer_id]

			agents_alive = agents.alive.at[childs_buffer_id].set(childs_alive) #make sur to not overwrite occupied buffer ids (if more reproducers than free buffer spots)
			
			agents_prms = agents.prms.at[childs_buffer_id].set(childs_prms)
			
			agents_policy_states = jax.tree.map(lambda x, c: x.at[childs_buffer_id].set(c), agents.policy_state, childs_policy_states)
			agents_energy = agents.energy.at[childs_buffer_id].set(childs_energy)
			agents_energy = agents_energy.at[parents_buffer_id].add(-self.reproduction_energy_cost * childs_mask)
			
			agents_positions = agents.position.at[childs_buffer_id].set(childs_positions)
			
			agents_tat = agents.time_above_threshold
			agents_tbt = agents.time_below_threshold
			agents_tat = agents_tat.at[parents_buffer_id].set(0)
			agents_tat = agents_tat.at[childs_buffer_id].set(0)
			
			agents_tbt = agents_tbt.at[childs_buffer_id].set(0)

			agents_age = agents.age.at[childs_buffer_id].set(0)

			agents_reward = agents.reward.at[childs_buffer_id].set(0)

			agents_reproduce = agents.reproduce.at[childs_buffer_id].set(False)

			agents_n_offsprings = agents.n_offsprings.at[childs_buffer_id].set(0)
			agents_n_offsprings = agents_n_offsprings.at[parents_buffer_id].add(1)

			childs_ids = jnp.where(childs_mask, jnp.cumsum(childs_mask, dtype=ui32)+state.last_agent_id+1, 0)
			agents_id = agents.id_.at[childs_buffer_id].set(childs_ids)

			childs_parent_id = agents.id_[parents_buffer_id]
			agents_parent_ids = agents.parent_id_.at[childs_buffer_id].set(childs_parent_id)

			parents_generation = agents.generation[parents_buffer_id]
			agents_generation = agents.generation.at[childs_buffer_id].set(parents_generation+1)

			agents = Agent(
				prms 				 = agents_prms,
				policy_state 		 = agents_policy_states,
				# ---
				alive 				 = agents_alive,
				age 				 = agents_age,
				position 			 = agents_positions,
				energy  			 = agents_energy,
				time_above_threshold = agents_tat,
				time_below_threshold = agents_tbt,
				# ---
				reward 				 = agents_reward,
				reproduce 			 = agents_reproduce,
				# ---
				id_ 				 = agents_id,
				parent_id_ 			 = agents_parent_ids,
				generation 			 = agents_generation,
				n_offsprings 		 = agents_n_offsprings
			)
			return agents
		# ---	

		reproducing = agents.reproduce # N,

		agents = jax.lax.cond(
			jnp.any(reproducing)&jnp.any(~agents.alive),
			_reproduce, 
			lambda _, agents: agents, 
			reproducing, agents
		)

		agents = agents._replace(
			age = jnp.where(agents.alive, agents.age+1, 0)
		)

		state = state._replace(
			agents=agents, 
			last_agent_id=agents.id_.max(),
		)

		return state, dict(reproducing=reproducing, dying=dead, avg_dead_age=avg_dead_age)

	# ---

	def _get_observations(self, state: EnvState)->Observation:
		"""
		returns agents observations
		"""
		chemical_fields = jnp.sum(state.food[:,None] * self.food_types.chemical_signature[...,None,None], axis=0)
		chemical_fields = self.chemicals_diffusion_conv(chemical_fields)

		agents = state.agents
		agents_i, agents_j = agents.position.T
		agents_alive_grid = jnp.zeros(self.size).at[agents_i, agents_j].add(agents.alive)
		agents_scent_field = jsp.signal.convolve(agents_alive_grid, self.agent_scent_diffusion_kernel, method="fft", mode="same")
		
		chemical_fields = jnp.concatenate([agents_scent_field[None], chemical_fields],axis=0)
		chemical_fields = jnp.where(chemical_fields<self.chemical_detection_threshold, 0.0, chemical_fields) #C,H,W

		agents_chemicals_inputs = jax.vmap(self.vision_fn, in_axes=(None,0))(chemical_fields, agents.position)

		agents_internal_inputs = jnp.concatenate([agents.energy[:,None], agents.reward[:,None]], axis=-1)

		agents_walls_inputs = jax.vmap(self.vision_fn, in_axes=(None,0))(self.walls[None], agents.position)

		return Observation(chemicals=agents_chemicals_inputs, internal=agents_internal_inputs, walls=agents_walls_inputs)

	# ---

	def _apply_actions(self, state: EnvState, actions: jax.Array)->Tuple[EnvState, dict]:
		"""
		"""
		agents = state.agents
		actions = jnp.where(agents.alive[:,None], actions, jnp.zeros(2, dtype=jnp.int16))

		# # --- 1. Predation ---

		# if self.predation:
		# 	attacking = (actions == 7)&(agents.alive)
		# 	i, j = agents.position.T
		# 	on_attacking_cell = (jnp.zeros(self.size, dtype=jnp.bool).at[i,j].set(attacking))[i,j]
		# 	attacked = (~attacking) & (agents.alive) & (on_attacking_cell) 
		# 	successfull_attacks = attacking & (jnp.zeros(self.size, dtype=jnp.bool).at[i,j].set(attacked))[i,j]

		# 	agents_alive = jnp.where(attacked, False, agents.alive)
		# 	agents_energy = jnp.where(attacking, agents.energy-self.predation_energy_cost, agents.energy)
		# 	agents_energy = jnp.where(successfull_attacks, agents_energy+self.predation_energy_gain, agents_energy)
		# 	agents = agents._replace(energy=agents_energy, alive=agents_alive)

		# --- 2. Move ---

		new_positions = jnp.mod(agents.position+actions, jnp.array(self.size, dtype=i16)[None])
		hits_wall = self.walls[*new_positions.T].astype(bool) #type:ignore
		
		# --- update energy
		positions = jnp.where(hits_wall[:,None], agents.position, new_positions)
		agents = agents._replace(position=positions)

		if self.deadly_walls:
			agents = agents._replace(alive=agents.alive&(~hits_wall))

		# --- 3. Eat ---
		# Agents can eat if:
		# 	always if self.passive_eating is True
		#	eating action (5) is taken

		eating_agents = (actions==5)&(agents.alive) if not self.passive_eating else agents.alive
		
		food = state.food
		agents_i, agents_j = agents.position.T
		eating_agents_grid = jnp.zeros(self.size, dtype=ui8).at[agents_i,agents_j].add(eating_agents.astype(jnp.uint8)) #nb of eating agents in each cell
		energy_grid = jnp.sum(food*self.food_types.energy_concentration[:,None,None], axis=0) #total qty of energy in each cell
		energy_intake_per_agent = jnp.where(eating_agents_grid>0, energy_grid/eating_agents_grid, 0.0)
		agents_energy_intake = jnp.where(agents.alive, energy_intake_per_agent[agents_i, agents_j], 0.0)

		agents_energy = jnp.clip(agents.energy + agents_energy_intake, -jnp.inf, self.max_energy)

		agents = agents._replace(energy=agents_energy)
		food = jnp.where(eating_agents_grid[None]>0, False, food)

		# --- 4. Reproduce ---
		# agents reproduce if :
		# 	reproduce action is taken if passive reproduction is false
		#	Their energy level has been above threshold for enough time
		# ---
		reproduce = agents.alive & (actions==6) if not self.passive_reproduction else agents.alive
		reproduce = reproduce & (agents.time_above_threshold > self.time_above_threshold_to_reproduce)
		agents = agents._replace(reproduce=reproduce)

		return (
			state._replace(agents=agents, food=food), 
			{"energy_intakes":agents_energy_intake}
		)

	# ====================== FOOD =========================

	@property
	def n_food_types(self):
		return self.food_types.growth_rate.shape[0]

	def _init_food(self, key)->FoodMap:
		food = jr.bernoulli(key, self.food_types.initial_density[:,None,None], (self.n_food_types, *self.size)).astype(ui8)
		food = jnp.where(jnp.cumsum(food,axis=0)>1, 0, food)
		return food

	# ---

	def _update_food(self, state: EnvState, key: jax.Array):
		"""Do one step of food growth"""
		food = state.food
		# --- Grow ---
		p_grow = self.growth_conv(food); assert isinstance(p_grow, jax.Array)
		p_grow = jnp.where(jnp.any(food, axis=0, keepdims=True), 0.0, p_grow)
		grow = jr.bernoulli(key, p_grow)

		i, j = state.agents.position.T
		agents_grid = jnp.zeros(self.size, dtype=bool).at[i,j].set(state.agents.alive)
		grow = jnp.where(
			jnp.cumsum(grow.astype(jnp.uint4),axis=0)>1 | self.walls[None] | agents_grid[None],
			False,
			grow
		)

		food = food | grow

		return state._replace(food=food)

	# ====================== RENDER =========================


	def render(self, state: EnvState, ax:plt.Axes|None=None):

		if ax is None:
			ax = plt.figure().add_subplot()
		else:
			ax=ax
		assert ax is not None

		food = state.food # F, X, Y
		F, H, W = food.shape
		agents = state.agents
		food_colors = plt.cm.Set2(jnp.arange(food.shape[0])) #type:ignore

		img = jnp.ones((F,H,W,4)) * food_colors[:,None,None]
		img = jnp.clip(jnp.where(food[...,None], img, 0.).sum(0), 0.0, 1.0) #type:ignore
		img = img.at[:,:,-1].set(jnp.any(food, axis=0))

		img = jnp.where(self.walls[...,None], jnp.array([0.5, 0.5, 0.5, 1.0]), img)

		ai, aj = agents.position[agents.alive].T
		img = img.at[ai,aj].set(jnp.array([0.,0.,0.,1.]))

		ax.imshow(img)

	# ---

	def render_states(self, states: list|EnvState, ax: plt.Axes, cam: Camera):

		if isinstance(states, EnvState):
			T = states.time.shape[0]
			states = [jax.tree.map(lambda x:x[t], states) for t in range(T)]

		for state in states:
			self.render(state, ax)
			cam.snap()

		return cam


#=======================================================================
#								INTERFACE
#=======================================================================

