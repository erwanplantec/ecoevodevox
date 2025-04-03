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
from jaxtyping import Array, Bool, Float, Float16, Int, Int16, Int32, Int64, Int8, PyTree

# ======================== UTILS =============================

from .utils import *

type FoodMap = Int[Array, "F H W"]
type KeyArray = jax.Array
type AgentState = PyTree
type AgentParams = jax.Array
type Action = jax.Array

# ============================================================

class Agent(NamedTuple):
	# ---
	policy_state: PyTree
	# ---
	alive: Bool
	age: Int16
	position: Int16
	energy: Float16
	reward: Int8
	reproduce: Bool
	time_above_threshold: Int16
	time_below_threshold: Int16
	n_offsprings: Int16
	id_: Int64
	parent_id_: Int64
	# ---
	prms: PyTree

class ChemicalType(NamedTuple):
	diffusion_rate: Float16

class FoodType(NamedTuple):
	reproduction_rate: Float16
	expansion_rate: Float16
	max_concentration: Int16
	chemical_signature: Float16
	energy_concentration: Float16
	spontaneous_grow_prob: Float16

_food_types_types = FoodType(f16, f16, i16, f16, f16, f16)

class EnvState(NamedTuple):
	agents: Agent
	food: FoodMap
	time: Int32
	last_agent_id: Int64=0

class Observation(NamedTuple):
	chemicals: jax.Array
	internal: jax.Array
	walls: jax.Array

class GridWorld:
	# ---
	def __init__(
		self, 
		size: Tuple[int, int],
		agent_fctry: Callable[[KeyArray], AgentParams], 
		agent_init: Callable[[AgentParams, KeyArray], AgentState],
		agent_apply: Callable[[AgentParams, Observation, AgentState, KeyArray], Action],
		mutation_fn: Callable[[AgentParams,KeyArray],AgentParams], 
		chemical_types: ChemicalType,
		food_types: FoodType,  
		max_agents: int=1_024, 
		init_agents: int=256,
		passive_eating: bool=True,
		passive_reproduction: bool=True,
		predation: bool=False,
		max_age: int=1_000,
		field_of_view: int=1,
		birth_pool_size: int|None=None,
		energy_reproduction_threshold: float=0.,
		reproduction_energy_cost: float=0.5,
		predation_energy_gain: float=5.,
		predation_energy_cost: float=0.1,
		move_energy_cost: float=0.1,
		base_energy_loss: float=0.05,
		max_energy: float=10.0,
		time_above_threshold_to_reproduce: int=20,
		time_below_threshold_to_die: int=10,
		initial_agent_energy: float=1.0,
		initial_food_density: Float=0.01,
		chemical_detection_threshold: float=0.01,
		deadly_walls: bool=False,
		size_apply_minibatches: int|None=None,
		size_init_minibatches: int|None=None):
		
		self.size = size
		self.walls = jnp.pad(jnp.zeros([s-2 for s in self.size], dtype=f16), 1, constant_values=1)
		self.deadly_walls = deadly_walls

		self.food_types = jax.tree.map(lambda x, ty: x.astype(ty), food_types, _food_types_types)
		self.nb_food_types = food_types.reproduction_rate.shape[0]
		self.initial_food_density = jnp.full((self.nb_food_types,), initial_food_density)
		growth_kernels = jnp.stack([jnp.array([[0.0, r/4, 0.0],
					       					   [r/4, 1-r, r/4],
					       					   [0.0, r/4, 0.0]], dtype=f16) for r in self.food_types.expansion_rate])
		self.food_growth_kernels = growth_kernels * self.food_types.reproduction_rate[:,None,None]
	
		self.chemical_types = chemical_types
		sx, sy = self.size
		norms = jnp.linalg.norm(jnp.mgrid[-sx//2:sx//2, -sy//2:sy//2], axis=0)
		diffusion_rates = self.chemical_types.diffusion_rate
		self.chemicals_diffusion_kernels = jnp.stack([jnp.exp(-norms/r) for r in diffusion_rates])

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
		self.move_energy_cost = move_energy_cost
		self.predation_energy_gain = predation_energy_gain
		self.predation_energy_cost = predation_energy_cost
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
		state = eqx.tree_at(lambda s: s.agents.policy_state, state, policy_states)
		state, actions_data = self._apply_actions(state, actions)

		# --- 3. Die / reproduce ---
		state, update_agents_data = self._update_agents(state, key)

		state = eqx.tree_at(lambda s: s.agents.age, state, state.agents.age+state.agents.alive)
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
		ids = jnp.where(alive, jnp.cumsum(alive, dtype=i64)+1, 0)
		prms = jax.vmap(self.agent_fctry)(jr.split(key_prms,self.max_agents))
		policy_states = self.mapped_agent_init(prms, jr.split(key_init, alive.shape[0]), alive)
		return Agent(
			alive=alive, 
			prms=prms, 
			energy=jnp.full((self.max_agents), self.initial_agent_energy, dtype=f16)*alive,
			reproduce=jnp.full((self.max_agents,), False, dtype=bool), 
			time_above_threshold=jnp.full((self.max_agents,), 0, dtype=i16), 
			time_below_threshold=jnp.full((self.max_agents,), 0, dtype=i16),
			position=jr.randint(key_pos, (self.max_agents, 2), minval=1, maxval=jnp.array(self.size, dtype=i16)-1, dtype=i16), 
			policy_state=policy_states, 
			reward=jnp.zeros((self.max_agents,), dtype=f16), 
			age=jnp.zeros((self.max_agents), dtype=i16), 
			n_offsprings=jnp.zeros(self.max_agents, dtype=i16),
			id_=ids,
			parent_id_=jnp.zeros(self.max_agents, dtype=ui32))

	# ---

	def _update_agents(self, state: EnvState, key: jax.Array)->Tuple[EnvState, PyTree]:
		
		key_repr, key_mut, key_init = jr.split(key, 3)
		agents = state.agents
		
		below_threshold = agents.energy < 0.
		above_threshold = agents.energy > self.energy_reproduction_threshold
		
		agents_tat = jnp.where(above_threshold&agents.alive, agents.time_above_threshold+1, 0); assert isinstance(agents_tat, jax.Array)
		agents_tbt = jnp.where(below_threshold&agents.alive, agents.time_below_threshold+1, 0); assert isinstance(agents_tbt, jax.Array)

		# --- 1. Death ---

		dead = (agents_tbt > self.time_below_threshold_to_die) | (agents.age > self.max_age)
		dead = dead & agents.alive
		agents_alive = agents.alive & ( ~dead )
		agents_age = jnp.where(agents_alive, agents.age, 0)
		agents = agents._replace(alive=agents_alive, time_above_threshold=agents_tat, time_below_threshold=agents_tbt, age=agents_age)

		# --- 2. Reproduce ---

		# ---

		def _reproduce(reproducing, agents):
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
			agents_age = jnp.where(agents_alive, agents.age+1, 0)

			agents_n_offsprings = agents.n_offsprings.at[childs_buffer_id].set(0)
			agents_n_offsprings = agents_n_offsprings.at[parents_buffer_id].add(1)

			childs_ids = jnp.where(childs_mask, jnp.cumsum(childs_mask, dtype=i64)+state.last_agent_id+1, 0)
			agents_id = agents.id_.at[childs_buffer_id].set(childs_ids)

			childs_parent_id = agents.id_[parents_buffer_id]
			parent_ids = agents.parent_id_.at[childs_buffer_id].set(childs_parent_id)

			agents = agents._replace(
				alive=agents_alive, 
				prms=agents_prms, 
				policy_state=agents_policy_states, 
				energy=jnp.clip(agents_energy, -jnp.inf, self.max_energy), 
				position=agents_positions, 
				time_above_threshold=agents_tat, 
				time_below_threshold=agents_tbt, 
				age=agents_age,
				n_offsprings=agents_n_offsprings,
				id_=agents_id,
				parent_id_=parent_ids
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

		new_last_id_ = agents.id_.max()

		state = state._replace(agents=agents, last_agent_id=new_last_id_)
		return state, dict(reproducing=reproducing, dying=dead)

	# ---

	def _get_observations(self, state: EnvState)->Observation:
		"""
		returns agents observations
		"""
		chemical_fields = jnp.sum(state.food[:,None] * self.food_types.chemical_signature[...,None,None], axis=0)
		chemical_fields = jax.vmap(lambda x, k: jsp.signal.convolve(x, k, mode="same", method="fft"))(chemical_fields, self.chemicals_diffusion_kernels)

		agents = state.agents
		agents_i, agents_j = agents.position.T
		agents_alive_grid = jnp.zeros(self.size).at[agents_i, agents_j].add(agents.alive)
		agents_scent_field = jsp.signal.convolve(agents_alive_grid, self.agent_scent_diffusion_kernel, method="fft", mode="same")
		
		chemical_fields = jnp.concatenate([agents_scent_field[None], chemical_fields],axis=0)
		chemical_fields = jnp.where(chemical_fields<self.chemical_detection_threshold, 0.0, chemical_fields) #C,H,W

		fov = self.field_of_view
		padded_chemical_fields = jnp.pad(chemical_fields, [(0,0),(fov,fov),(fov,fov)])
		agents_chemicals_inputs = jax.vmap(partial(k_neighborhood, k=fov), in_axes=(None,0,0))(padded_chemical_fields, agents_i+fov, agents_j+fov)

		agents_internal_inputs = jnp.concatenate([agents.energy[:,None], agents.reward[:,None]], axis=-1)

		agents_walls_inputs = jax.vmap(moore_neighborhood, in_axes=(None,0,0))(self.walls[None], agents_i, agents_j)

		return Observation(chemicals=agents_chemicals_inputs, internal=agents_internal_inputs, walls=agents_walls_inputs)

	# ---

	def _apply_actions(self, state: EnvState, actions: jax.Array)->Tuple[EnvState, dict]:
		"""
		"""
		agents = state.agents
		actions = jnp.where(agents.alive[:,None], actions, jnp.zeros(2, dtype=jnp.int16))
		no_move = jnp.all(actions == jnp.zeros(2, dtype=jnp.int16)[None], axis=-1)

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

		new_positions = agents.position+actions
		hits_wall = self.walls[*new_positions.T].astype(bool) #type:ignore
		positions = jnp.where(hits_wall[:,None], agents.position, new_positions)
		energy_loss = jnp.where(no_move, self.base_energy_loss, self.base_energy_loss+self.move_energy_cost)
		agents_energy = jnp.where(agents.alive, agents.energy-energy_loss, 0.0)
		agents = agents._replace(position=positions, energy=agents_energy)

		if self.deadly_walls:
			agents = agents._replace(alive=agents.alive&(~hits_wall))

		# --- 3. Eat ---
		# Agents can eat if:
		# 	always if self.passive_eating is True
		#	eating action (5) is taken

		eating_agents = (actions==5)&(agents.alive) if not self.passive_eating else agents.alive
		
		food = state.food
		agents_i, agents_j = agents.position.T
		eating_agents_grid = jnp.zeros(self.size, dtype=i16).at[agents_i,agents_j].add(eating_agents) #nb of eating agents in each cell
		energy_grid = jnp.sum(food*self.food_types.energy_concentration[:,None,None], axis=0) #total qty of energy in each cell
		energy_intake_per_agent = jnp.where(eating_agents_grid>0, energy_grid/eating_agents_grid, 0.0)
		agents_energy_intake = jnp.where(agents.alive, energy_intake_per_agent[agents_i, agents_j], 0.0)

		agents_energy = jnp.clip(agents.energy + agents_energy_intake, -jnp.inf, self.max_energy)

		agents = agents._replace(energy=agents_energy)
		food = jnp.clip(food-eating_agents_grid[None], 0, self.food_types.max_concentration[:,None,None])

		# --- 4. Reproduce ---
		# agents reproduce if :
		# 	reproduce action is taken if passive reproduction is false
		#	Their energy level has been above threshold for enough time
		# ---
		reproduce = agents.alive & (actions==6) if not self.passive_reproduction else agents.alive
		reproduce = reproduce & (agents.time_above_threshold > self.time_above_threshold_to_reproduce)
		agents = agents._replace(reproduce=reproduce)

		return state._replace(agents=agents, food=food), {"energy_intakes":agents_energy_intake}

	# ====================== FOOD =========================

	@property
	def n_food_types(self):
		return self.food_types.reproduction_rate.shape[0]

	def _init_food(self, key)->FoodMap:
		food = jr.bernoulli(key, self.initial_food_density[:,None,None], (self.n_food_types, *self.size)).astype(i16)
		food = jnp.where(jnp.cumsum(jnp.pad(food,((1,0),(0,0),(0,0)))[:-1],axis=0)>0, 0, food)
		return food

	# ---

	def _update_food(self, state: EnvState, key: jax.Array):
		"""Do one step of food growth"""
		food = state.food

		# --- Grow ---
		p_grow = jax.vmap(partial(jsp.signal.convolve2d, mode="same"))(food, self.food_growth_kernels)
		agents_i, agents_j = state.agents.position.T
		agents_grid = jnp.zeros(self.size, dtype=jnp.int16).at[agents_i,agents_j].add(state.agents.alive.astype(jnp.int16)) > 0
		p_grow = jnp.where(agents_grid, 0.0, p_grow)
		grow = jr.bernoulli(key, p_grow).astype(i16)
		grow = jnp.where(jnp.cumsum(jnp.pad(food,((1,0),(0,0),(0,0)))[:-1],axis=0)>0, 0, grow)
		food = jnp.clip(food + grow, 0, self.food_types.max_concentration[:,None,None])

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
		img = img.at[:,:,-1].set((food/self.food_types.max_concentration.max()).sum(0))

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



