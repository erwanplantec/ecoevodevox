from functools import partial
import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import jax.scipy as jsp
import equinox as eqx
import equinox.nn as nn
from celluloid import Camera

import matplotlib.pyplot as plt

from typing import Callable, NamedTuple, Tuple, TypeAlias
from jaxtyping import Bool, Float, Float16, Int, Int16, Int32, Int64, Int8, PyTree

from src.eco.utils import minivmap

f16, f32, i8, i16, i32, i64 = jnp.float16, jnp.float32, jnp.uint8, jnp.uint16, jnp.uint32, jnp.uint64
MAX_INT16 = jnp.iinfo(jnp.uint16).max
boolean_maxpool = lambda x: nn.Pool(init=False, operation=jnp.logical_or, num_spatial_dims=2, padding=1, kernel_size=3)(x[None])[0]
convolve = partial(jsp.signal.convolve, mode="same")

def neighbor_states_fn(x, include_center=True, neighborhood="moore"):
	if x.ndim>2:
		extra_offs = (0,)*(x.ndim-2)
	else:
		extra_offs = ()
	if neighborhood=="moore":
		shifts = [(*extra_offs, 0,1), (*extra_offs, 1,0), (*extra_offs, 0,-1), (*extra_offs, -1,0)]
	elif neighborhood=="vn":
		shifts = [(*extra_offs, di, dj) for di in [-1,0,1] for dj in [-1,0,1]]
	else:
		raise ValueError(f"neighborhood {neighborhood} is not valid. must be either 'moore' or 'vn'")
	output = jnp.stack([jnp.roll(x, shift) for shift in shifts])
	if include_center:
		output = jnp.concatenate([x[None], output], axis=0)
	return output

def moore_neighborhood(x, i, j):
	C, H, W = x.shape
	return jax.lax.dynamic_slice(x, [jnp.array(0,dtype=i16),i,j], [C,3,3])

class Agent(NamedTuple):
	# ---
	policy_state: PyTree
	# ---
	alive: Bool
	age: Int16
	position: Int16
	energy: Float16
	reward: Int8
	time_above_threshold: Int16
	time_below_threshold: Int16
	n_offsprings: Int16
	id_: Int64
	parent_id_: Int64
	# ---
	prms: PyTree

class FoodType(NamedTuple):
	reproduction_rate: Float16
	diffusion_rate: Float16
	energy_concentration: Float16

FoodMap: TypeAlias=jax.Array

class EnvState(NamedTuple):
	agents: Agent
	food: FoodMap
	time: Int32
	last_agent_id: Int64=0

class Observation(NamedTuple):
	chemicals: jax.Array
	internal: jax.Array
	walls: jax.Array

KeyArray: TypeAlias = jax.Array
AgentState: TypeAlias = PyTree
AgentParams: TypeAlias = jax.Array
Action: TypeAlias = jax.Array

class GridWorld:
	# ---
	def __init__(
		self, 
		agent_fctry: Callable[[KeyArray], AgentParams], 
		agent_init: Callable[[AgentParams, KeyArray], AgentState],
		agent_apply: Callable[[AgentParams, Observation, AgentState, KeyArray], Action],
		food_types: FoodType, 
		size: Tuple[int, int], 
		mutation_fn: Callable[[AgentParams,KeyArray],AgentParams], 
		max_agents: int=1_024, 
		init_agents: int=256,
		predation: bool=True,
		max_age: int=1_000,
		birth_pool_size: int|None=None,
		energy_reproduction_threshold: float=0.,
		reproduction_energy_cost: float=0.5,
		move_energy_cost: float=0.1,
		base_energy_loss: float=0.1,
		time_above_threshold_to_reproduce: int=20,
		time_below_threshold_to_die: int=10,
		initial_agent_energy: float=1.0,
		agent_scent_diffusion: float=1.0,
		initial_food_density: float=0.01,
		spontaneous_grow_prob: float=0.0,
		chemical_detection_threshold: float=0.01,
		deadly_walls: bool=False,
		perception_neighborhood: str="moore",
		size_apply_minibatches: int|None=None,
		size_init_minibatches: int|None=None):
		
		self.size = size
		self.walls = jnp.pad(jnp.zeros([s-2 for s in self.size], dtype=f16), 1, constant_values=1)
		self.deadly_walls = deadly_walls
		self.perception_neighborhood = perception_neighborhood

		self.food_types = jax.tree.map(lambda x: x.astype(f16), food_types)
		self.nb_food_types = food_types.diffusion_rate.shape[0]
		self.initial_food_density = initial_food_density
		self.spontaneous_grow_prob = spontaneous_grow_prob
		sx, sy = self.size
		norms = jnp.linalg.norm(jnp.mgrid[-sx//2:sx//2, -sy//2:sy//2], axis=0)
		diffusion_rates = food_types.diffusion_rate
		self.chemicals_diffusion_kernels = jnp.stack([jnp.exp(-norms/r) for r in diffusion_rates])

		self.mutation_fn = mutation_fn
		self.agent_fctry = agent_fctry
		self.agent_init = agent_init
		self.agent_apply = agent_apply
		
		dummy_policy_state = agent_init(agent_fctry(jr.key(1)), jr.key(1))
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
		self.energy_reproduction_threshold = energy_reproduction_threshold
		self.time_above_threshold_to_reproduce = time_above_threshold_to_reproduce
		self.time_below_threshold_to_die = time_below_threshold_to_die
		self.initial_agent_energy = initial_agent_energy
		self.reproduction_energy_cost = reproduction_energy_cost
		self.base_energy_loss = base_energy_loss
		self.agent_scent_diffusion_kernel = jnp.exp(-norms/agent_scent_diffusion**2)
		self.chemical_detection_threshold = chemical_detection_threshold
		self.move_energy_cost = move_energy_cost
		self.max_age = max_age

		self.n_actions = 6 

	# ---

	def step(self, state: EnvState, key: jax.Array)->Tuple[EnvState,PyTree]:
		
		# --- 1. Update food sources ---
		key, key_food = jr.split(key)
		state = self._update_food(state, key_food)

		# --- 2. Get and apply actions ---
		key, key_action = jr.split(key)
		observations = self._get_observations(state)
		actions, policy_states = self.mapped_agent_apply(state.agents.prms, observations, state.agents.policy_state, jr.split(key_action, self.max_agents), state.agents.alive)
		state = eqx.tree_at(lambda s: s.agents.policy_state, state, policy_states)
		state = self._apply_actions(state, actions)

		# --- 3. Die / reproduce ---
		state = self._update_agents(state, key)

		state = eqx.tree_at(lambda s: s.agents.age, state, state.agents.age+state.agents.alive)
		state = state._replace(time=state.time+1)

		return state, dict(state=state, actions=actions, observations=observations)

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
			time_above_threshold=jnp.full((self.max_agents,), 0, dtype=i8), 
			time_below_threshold=jnp.full((self.max_agents,), 0, dtype=i8),
			position=jr.randint(key_pos, (self.max_agents, 2), minval=1, maxval=jnp.array(self.size, dtype=i16)-1, dtype=i16), 
			policy_state=policy_states, 
			reward=jnp.zeros((self.max_agents,), dtype=f16), 
			age=jr.randint(key_age, (self.max_agents,), minval=1, maxval=self.max_age), 
			n_offsprings=jnp.zeros(self.max_agents, dtype=i16),
			id_=ids,
			parent_id_=jnp.zeros(self.max_agents, dtype=i64))

	# ---

	def _update_agents(self, state: EnvState, key: jax.Array)->EnvState:
		
		key_repr, key_mut, key_init = jr.split(key, 3)
		agents = state.agents
		agents_energy = agents.energy
		below_threshold = agents_energy < 0.
		above_threshold = agents_energy > self.energy_reproduction_threshold
		
		agents_tat = (agents.time_above_threshold + above_threshold) * above_threshold
		agents_tbt = (agents.time_below_threshold + below_threshold) * below_threshold

		# --- 1. Death ---

		dead = (agents_tbt > self.time_below_threshold_to_die) | (agents.age > self.max_age)
		agents_alive = agents.alive & ( ~dead )

		# --- 2. Reproduce ---
		reproducing = (agents_tat > self.time_above_threshold_to_reproduce) # N,
		free_buffer_spots = ~agents_alive # N,
		_, parents_buffer_id = jax.lax.top_k(reproducing+jr.uniform(key_repr,reproducing.shape,minval=-0.1,maxval=0.1), self.birth_pool_size)
		parents_mask = reproducing[parents_buffer_id]
		parents_prms = agents.prms[parents_buffer_id]
		is_free, childs_buffer_id = jax.lax.top_k(free_buffer_spots, self.birth_pool_size)
		childs_mask = parents_mask & is_free #is a child if parent was actually reproducing and there are free buffer spots

		childs_alive = childs_mask
		childs_prms = jax.vmap(self.mutation_fn)(parents_prms, jr.split(key_mut, self.birth_pool_size))
		childs_policy_states = self.mapped_agent_init(childs_prms, jr.split(key_init, self.birth_pool_size), childs_mask)
		childs_energy = jnp.full(self.birth_pool_size, self.initial_agent_energy, dtype=f16)
		childs_positions = agents.position[parents_buffer_id]

		agents_alive = agents_alive.at[childs_buffer_id].set(
			jnp.where(childs_mask, childs_alive, agents_alive[childs_buffer_id])
		) #make sur to not overwrite occupied buffer ids (if more reproducers than free buffer spots)
		
		agents_prms = agents.prms.at[childs_buffer_id].set(
			jnp.where(childs_mask[:,None], childs_prms, agents.prms[childs_buffer_id])
		)
		
		agents_policy_states = jax.tree.map(
			lambda x, cs, os: x.at[childs_buffer_id].set(
				jnp.where(jnp.expand_dims(childs_mask, [i+1 for i in range(cs.ndim-1)]), cs, os[childs_buffer_id])
			), 
			agents.policy_state, childs_policy_states, agents.policy_state
		)
		agents_energy = agents_energy.at[childs_buffer_id].set(
			jnp.where(childs_mask, childs_energy, agents_energy[childs_buffer_id])
		)
		agents_energy = agents_energy.at[parents_buffer_id].add(-self.reproduction_energy_cost * childs_mask)
		
		agents_positions = agents.position.at[childs_buffer_id].set(
			jnp.where(childs_mask[:,None], childs_positions, agents.position[childs_buffer_id])
		)
		
		agents_tat = agents_tat.at[parents_buffer_id].set(jnp.where(childs_mask, 0, agents_tat[parents_buffer_id]))
		agents_tat = agents_tat.at[childs_buffer_id].set(jnp.where(childs_mask, 0, agents_tat[childs_buffer_id]))
		
		agents_tbt = agents_tbt.at[childs_buffer_id].set(jnp.where(childs_mask, 0, agents_tbt[childs_buffer_id]))

		agents_age = agents.age.at[childs_buffer_id].set(jnp.where(childs_mask, 0, agents.age[childs_buffer_id]))
		agents_age = jnp.where(agents_alive, agents.age+1, 0)

		agents_n_offsprings = agents.n_offsprings.at[childs_buffer_id].set(jnp.where(childs_mask,0,agents.n_offsprings[childs_buffer_id]))
		agents_n_offsprings = agents_n_offsprings.at[parents_buffer_id].add(1)

		childs_ids = jnp.where(childs_mask, jnp.cumsum(childs_mask, dtype=i64)+state.last_agent_id+1, 0)
		agents_id = agents.id_.at[childs_buffer_id].set(
			jnp.where(childs_mask, childs_ids, agents.id_[childs_buffer_id]) #type:ignore
		)

		childs_parent_id = agents.id_[parents_buffer_id]
		parent_ids = agents.parent_id_.at[childs_buffer_id].set(
			jnp.where(childs_mask, childs_parent_id, agents.id_[childs_buffer_id])
		)

		agents = agents._replace(
			alive=agents_alive, 
			prms=agents_prms, 
			policy_state=agents_policy_states, 
			energy=agents_energy, 
			position=agents_positions, 
			time_above_threshold=agents_tat, 
			time_below_threshold=agents_tbt, 
			age=agents_age,
			n_offsprings=agents_n_offsprings,
			id_=agents_id,
			parent_id_=parent_ids
		)

		new_last_id_ = agents_id.max()

		return state._replace(agents=agents, last_agent_id=new_last_id_)

	# ---

	def _get_observations(self, state: EnvState)->Observation:
		"""
		returns agents observations
		"""
		chemical_fields = jax.vmap(lambda x, k: jsp.signal.convolve(x, k, mode="same", method="fft"))(state.food, self.chemicals_diffusion_kernels)

		agents = state.agents
		agents_i, agents_j = agents.position.T
		agents_alive_grid = jnp.zeros(self.size).at[agents_i, agents_j].set(1.)
		agents_scent_field = jsp.signal.convolve(agents_alive_grid, self.agent_scent_diffusion_kernel, method="fft", mode="same")
		
		chemical_fields = jnp.concatenate([agents_scent_field[None], chemical_fields],axis=0)
		chemical_fields = jnp.where(chemical_fields<self.chemical_detection_threshold, 0.0, chemical_fields) #C,H,W

		padded_chemical_fields = jnp.pad(chemical_fields, [(0,0),(1,1),(1,1)])
		agents_chemicals_inputs = jax.vmap(moore_neighborhood, in_axes=(None,0,0))(padded_chemical_fields, agents_i+1, agents_j+1)

		agents_internal_inputs = jnp.concatenate([agents.energy[:,None], agents.reward[:,None]], axis=-1)

		agents_walls_inputs = jax.vmap(moore_neighborhood, in_axes=(None,0,0))(self.walls[None], agents_i, agents_j)

		return Observation(chemicals=agents_chemicals_inputs, internal=agents_internal_inputs, walls=agents_walls_inputs)

	# ---

	def _apply_actions(self, state: EnvState, actions: jax.Array)->EnvState:
		"""
		actions are: {0:N, 1:E, 2:S, 3:W, 4:eat, 5:none}
		"""
		action_effects = jnp.array([[1,0],[0,1], [-1,0], [0,-1], [0,0], [0,0]], dtype=i16)
		agents = state.agents
		actions = jnp.where(agents.alive, actions, 4)
		agents_moves = action_effects[actions]
		new_positions = agents.position+agents_moves
		hits_wall = self.walls[*new_positions.T].astype(bool) #type:ignore
		positions = jnp.where(hits_wall[:,None], agents.position, new_positions)
		energy_loss = (actions < 4) * self.move_energy_cost + jnp.array(self.base_energy_loss, dtype=f16)
		agents = agents._replace(position=positions, energy=agents.energy-energy_loss)

		if self.deadly_walls:
			agents = agents._replace(alive=agents.alive&(~hits_wall))

		eating_agents = (actions==4)&(agents.alive)
		food = state.food
		agents_i, agents_j = agents.position.T

		food_on_cell = food[:,agents_i,agents_j] # F,N
		eating_agents_on_cell = (jnp.zeros(self.size, dtype=f16).at[agents_i, agents_j].add(eating_agents))[agents_i,agents_j] #N,
		energy_on_cell = jnp.sum(food_on_cell * self.food_types.energy_concentration[:,None], axis=0) #N,
		agents_reward = jnp.where(
			agents.alive, 
			energy_on_cell / jnp.clip(eating_agents_on_cell, 1), 
			0.0
		)
		agents_energy = agents.energy + agents_reward
		
		agents = agents._replace(reward=agents_reward, energy=agents_energy)
		eating_agents_grid = jnp.zeros(self.size, dtype=bool).at[agents_i, agents_j].set(eating_agents)
		food = jnp.where(eating_agents_grid[None], False, food)

		return state._replace(agents=agents, food=food)


	# ====================== FOOD =========================

	@property
	def n_food_types(self):
		return self.food_types.diffusion_rate.shape[0]

	def _init_food(self, key)->FoodMap:
		food = jr.bernoulli(key, self.initial_food_density, (self.n_food_types, *self.size))
		food = jnp.where(jnp.sum(food,axis=0)>1, False, food)
		return food

	# ---

	def _update_food(self, state: EnvState, key: jax.Array):
		"""Do one step of food growth"""
		food = state.food

		# --- Reproduce ---
		neighbors_count = jax.vmap(convolve, in_axes=(0,None))(food, jnp.array([[0,1,0],[1,0,1],[0,1,0]]))
		p_grow = neighbors_count * self.food_types.reproduction_rate[:,None,None]
		p_grow = p_grow + self.spontaneous_grow_prob
		p_grow = jnp.where(neighbors_count>3, 0., p_grow)

		grow = jr.bernoulli(key, p_grow) & (~jnp.any(food, axis=0, keepdims=True))#type:ignore

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

