from flax.struct import PyTreeNode
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import equinox as eqx
from celluloid import Camera

import matplotlib.pyplot as plt

from typing import Callable, Literal, NamedTuple, Tuple
from jaxtyping import (
	Float, PyTree, Array,
	Bool,
	Int16, Int32,
	UInt8, UInt16, UInt32,
	Float16, Float32
)

from ..agents.interface import AgentInterface, AgentState, Position

# ======================== UTILS =============================

from .utils import *

type FoodMap = Bool[Array, "F H W"]
type KeyArray = jax.Array
type AgentParams = jax.Array
type Action = jax.Array
type Info = dict

@jax.jit
def get_cell_index(pos: Float16):
	indices = jnp.floor(pos).astype(jnp.int16)
	return indices

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
	D = jnp.sum(jnp.square(L), axis=0, keepdims=True) #1,H,W

	diffusion_kernels = jnp.exp(-D/diffusion_rates[:,None,None]); assert isinstance(diffusion_kernels, jax.Array)
	diffusion_kernels_fft = jnp.fft.fft2(jnp.fft.fftshift(diffusion_kernels, axes=(1,2)))

	@jax.jit
	def _conv(C: Float[jax.Array, "C H W"])->Float[jax.Array, "C H W"]:
		C_fft = jnp.fft.fft2(C)
		res = jnp.real(jnp.fft.ifft2(C_fft * diffusion_kernels_fft))
		return res

	return _conv


# ============================================================

class ChemicalType(PyTreeNode):
	diffusion_rate: Float16

class FoodType(PyTreeNode):
	growth_rate: Float32
	dmin: Float32
	dmax: Float32
	chemical_signature: Float32
	energy_concentration: Float32
	spontaneous_grow_prob: Float32
	initial_density: Float32

class EnvState(PyTreeNode):
	agents_states: AgentState
	food: FoodMap
	time: UInt32
	last_agent_id: UInt32=0

class Observation(PyTreeNode):
	chemicals: jax.Array
	internal: jax.Array
	walls: jax.Array

#=======================================================================

class GridworldConfig(PyTreeNode):
	# ---
	size: tuple[int,int]=(256,256)
	# ---
	walls_density: float=1e-4
	wall_effect: Literal["kill","penalize","none"]="kill"
	# ---
	max_agents: int=10_000
	init_agents: int=1_024
	max_age: int=1_000
	# ---
	reproduction_cost: float=0.5
	max_energy: float=50.0
	initial_energy: float=1.0
	time_above_threshold_to_reproduce: int=100
	time_below_threshold_to_die: int=30
	# ---
	chemicals_detection_threshold: float=1e-3
	# ---
	birth_pool_size: int=256
	# ---

#=======================================================================

class GridWorld:
	
	#-------------------------------------------------------------------

	def __init__(
		self, 
		cfg: GridworldConfig,
		# ---
		agent_interface: AgentInterface,
		mutation_fn: Callable[[AgentParams,jax.Array], AgentParams],
		# ---
		chemical_types: ChemicalType,
		food_types: FoodType, 
		# --- 
		*,
		key: jax.Array):
		
		self.cfg = cfg
		self.walls = jr.bernoulli(key, cfg.walls_density, cfg.size)

		self.food_types = jax.tree.map(lambda x: x.astype(jnp.float16), food_types)
		self.nb_food_types = food_types.growth_rate.shape[0]
		self.growth_conv = make_growth_convolution(cfg.size, 
												   food_types.growth_rate,
												   food_types.dmin, 
												   food_types.dmax)
	
		self.chemical_types = chemical_types
		self.chemicals_diffusion_conv = make_chemical_diffusion_convolution(cfg.size,
																			chemical_types.diffusion_rate)

		@jax.jit
		def _vision_fn(x: jax.Array, pos: Position):
			"""Return window of array corresponding to agent view if agent at pos"""
			indices = get_cell_index(agent_interface.full_body_pos(pos))
			return x[:, *indices] if x.ndim==3 else x[*indices]

		self.vision_fn = _vision_fn

		self.agent_interface = agent_interface
		self.mutation_fn = mutation_fn
		self.agent_scent_diffusion_kernel = jnp.array([[ 0.1 , 0.1 , 0.1 ],
													   [ 0.1 , 1.0 , 0.1 ],
													   [ 0.1 , 0.1 , 0.1 ]], dtype=f16)

	#-------------------------------------------------------------------

	def step(self, state: EnvState, key: jax.Array)->Tuple[EnvState,PyTree]:

		# --- 1. Update food sources ---
		key, key_food = jr.split(key)
		state = self._update_food(state, key_food)

		# --- 2. Get and apply actions ---
		key, key_step = jr.split(key)
		observations = self._get_observations(state)
		actions, agents_states, agents_step_data = jax.vmap(self.agent_interface.step)(
			observations, state.agents_states, jr.split(key_step, self.cfg.max_agents)
		)
		state = state.replace(agents_states=agents_states)
		state, actions_data = self._apply_actions(state, actions)

		# --- 3. Die / reproduce ---
		state, update_agents_data = self._update_agents(state, key)

		state = state.replace(time=state.time+1)

		return (
			state, 
			dict(state=state, 
				 actions=actions, 
				 observations=observations,
				 **agents_step_data,
				 **actions_data, 
				 **update_agents_data)
		)

	# ---

	def init(self, key: jax.Array)->EnvState:
		key_food, key_agent = jr.split(key, 2)
		food_sources = self._init_food(key_food)
		agents = self._init_agents(key_agent)
		return EnvState(agents_states=agents, food=food_sources, time=jnp.zeros((), dtype=jnp.uint32), last_agent_id=agents.id_.max())

	#-------------------------------------------------------------------

	def _init_agents(self, key):

		key_prms, key_pos, key_head, key_init = jr.split(key, 4)
		alive = jnp.arange(self.cfg.max_agents) < self.cfg.init_agents
		policy_params = jax.vmap(self.agent_interface.policy_fctry)(jr.split(key_prms,self.cfg.max_agents))
		policy_states, sensory_states, motor_states = jax.vmap(self.agent_interface.init)(
			policy_params, jr.split(key_init, self.cfg.max_agents)
		)
		positions = jr.uniform(key_pos, (self.cfg.max_agents, 2), minval=1.0, maxval=jnp.array(self.cfg.size, dtype=f16)-1, dtype=f16)
		headings = jr.uniform(key_head, (self.cfg.max_agents,), minval=0.0, maxval=2*jnp.pi, dtype=f16)
		positions = Position(positions, headings)

		return AgentState(
			# ---
			policy_params 		 = policy_params, 
			policy_state 		 = policy_states,
			sensory_state 		 = sensory_states,
			motor_state 		 = motor_states,
			# ---
			position 			 = positions, 
			# ---
			alive 				 = alive, 
			energy 				 = jnp.full((self.cfg.max_agents), self.cfg.initial_energy, dtype=f16)*alive, 
			time_above_threshold = jnp.full((self.cfg.max_agents,), 0, dtype=ui16), 
			time_below_threshold = jnp.full((self.cfg.max_agents,), 0, dtype=ui16),
			# ---
			reproduce 			 = jnp.full((self.cfg.max_agents,), False, dtype=bool),
			reward 				 = jnp.zeros((self.cfg.max_agents,), dtype=f16), 
			# ---
			age 				 = jnp.ones((self.cfg.max_agents), dtype=ui16), 
			n_offsprings 		 = jnp.zeros(self.cfg.max_agents, dtype=ui16),
			id_ 				 = jnp.where(alive, jnp.cumsum(alive, dtype=ui32), 0),
			parent_id_ 			 = jnp.zeros(self.cfg.max_agents, dtype=ui32),
			generation 			 = jnp.zeros(self.cfg.max_agents, dtype=ui16),
		)

	#-------------------------------------------------------------------

	def _update_agents(self, state: EnvState, key: jax.Array)->Tuple[EnvState, PyTree]:
		
		key_repr, key_mut, key_init = jr.split(key, 3)
		agents = state.agents_states
		
		below_threshold = agents.energy < 0.0
		above_threshold = ~below_threshold
		
		agents_tat = jnp.where(above_threshold&agents.alive, agents.time_above_threshold+1, 0); assert isinstance(agents_tat, jax.Array)
		agents_tbt = jnp.where(below_threshold&agents.alive, agents.time_below_threshold+1, 0); assert isinstance(agents_tbt, jax.Array)

		# --- 1. Death ---

		dead = (agents_tbt > self.cfg.time_below_threshold_to_die) | (agents.age > self.cfg.max_age)
		dead = dead & agents.alive
		avg_dead_age = jnp.where(dead, agents.age, 0).sum() / dead.sum() #type:ignore
		agents_alive = agents.alive & ( ~dead )
		agents_age = jnp.where(agents_alive, agents.age, 0)
		agents = agents.replace(alive=agents_alive, time_above_threshold=agents_tat, time_below_threshold=agents_tbt, age=agents_age)

		# --- 2. Reproduce ---

		# ---

		def _reproduce(reproducing: jax.Array, agents: AgentState, key: jax.Array)->AgentState:
			"""
			"""
			key_shuff, key_pos, key_head = jr.split(key, 3)

			free_buffer_spots = ~agents.alive # N,
			_, parents_buffer_id = jax.lax.top_k(reproducing+jr.uniform(key_shuff,reproducing.shape,minval=-0.1,maxval=0.1), self.cfg.birth_pool_size) # add random noise to have non deterministic sammpling
			parents_mask = reproducing[parents_buffer_id]
			parents_prms = jax.tree.map(lambda x: x[parents_buffer_id], agents.policy_params)
			is_free, childs_buffer_id = jax.lax.top_k(free_buffer_spots, self.cfg.birth_pool_size)
			childs_mask = parents_mask & is_free #is a child if parent was actually reproducing and there are free buffer spots
			childs_buffer_id = jnp.where(childs_mask, childs_buffer_id, self.cfg.max_agents) # assign wrong index if not born
			parents_buffer_id = jnp.where(childs_mask, parents_buffer_id, self.cfg.max_agents)

			childs_alive = childs_mask
			
			childs_policy_params = jax.vmap(self.mutation_fn)(parents_prms, jr.split(key_mut, self.cfg.birth_pool_size))
			childs_policy_states, childs_sensory_states, childs_motor_states = jax.vmap(self.agent_interface.init)(
				childs_policy_params, jr.split(key_init, self.cfg.birth_pool_size)
			)

			childs_policy_states = jax.vmap(self.agent_interface.policy_init)(childs_policy_params, jr.split(key_init, self.cfg.birth_pool_size))
			
			childs_energy = jnp.full(self.cfg.birth_pool_size, self.cfg.initial_energy, dtype=f16)

			childs_positions = agents.position.pos[parents_buffer_id] + jr.uniform(key_pos, minval=-1.0, maxval=1.0, dtype=f16)
			childs_headings = agents.position.heading[parents_buffer_id] + jr.uniform(key_head, minval=-1.0, maxval=1.0, dtype=f16)
			childs_headings = jnp.mod(childs_headings, 2*jnp.pi)
			childs_positions = Position(childs_positions, childs_headings)

			agents_alive = agents.alive.at[childs_buffer_id].set(childs_alive) #make sur to not overwrite occupied buffer ids (if more reproducers than free buffer spots)
			
			agents_policy_params = jax.tree.map(
				lambda x, x_child: x.at[childs_buffer_id].set(x_child),
				agents.policy_params, childs_policy_params
			)
			
			agents_policy_states = jax.tree.map(lambda x, c: x.at[childs_buffer_id].set(c), agents.policy_state, childs_policy_states)
			agents_sensory_states = jax.tree.map(lambda x, c: x.at[childs_buffer_id].set(c), agents.sensory_state, childs_sensory_states)
			agents_motor_states = jax.tree.map(lambda x, c: x.at[childs_buffer_id].set(c), agents.motor_state, childs_motor_states)

			agents_energy = agents.energy.at[childs_buffer_id].set(childs_energy)
			agents_energy = agents_energy.at[parents_buffer_id].add(-self.cfg.reproduction_cost * childs_mask)
			
			agents_positions = jax.tree.map(lambda x, c: x.at[childs_buffer_id].set(c), agents.position, childs_positions)
			
			agents_tat = agents.time_above_threshold
			agents_tbt = agents.time_below_threshold
			agents_tat = agents_tat.at[parents_buffer_id].set(0)
			agents_tat = agents_tat.at[childs_buffer_id].set(0)
			
			agents_tbt = agents_tbt.at[childs_buffer_id].set(0)

			agents_age = agents.age.at[childs_buffer_id].set(1)

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

			agents = AgentState(
				policy_params 		 = agents_policy_params,
				policy_state 		 = agents_policy_states,
				# ---
				sensory_state 		 = agents_sensory_states,
				motor_state 		 = agents_motor_states,
				# ---
				position 			 = agents_positions,
				# ---
				alive 				 = agents_alive,
				age 				 = agents_age,
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
				n_offsprings 		 = agents_n_offsprings,
			)
			return agents
		# ---	

		reproducing = agents.alive & (agents.time_above_threshold > self.cfg.time_above_threshold_to_reproduce)

		agents = jax.lax.cond(
			jnp.any(reproducing)&jnp.any(~agents.alive),
			_reproduce, 
			lambda repr, agts, key: agts, 
			reproducing, agents, key_repr
		)

		state = state.replace(
			agents_states=agents, 
			last_agent_id=agents.id_.max(),
		)

		return (
			state, 
			dict(reproducing=reproducing, dying=dead, avg_dead_age=avg_dead_age)
		)

	#-------------------------------------------------------------------

	def _get_observations(self, state: EnvState)->Observation:
		"""
		returns agents observations
		"""
		chemical_source_fields = jnp.sum(state.food[:,None] * self.food_types.chemical_signature[...,None,None], axis=0)
		chemical_fields = self.chemicals_diffusion_conv(chemical_source_fields)

		agents = state.agents_states
		agents_i, agents_j = get_cell_index(agents.position.pos).T
		agents_alive_grid = jnp.zeros(self.cfg.size).at[agents_i, agents_j].add(agents.alive)
		agents_scent_field = jsp.signal.convolve(agents_alive_grid, self.agent_scent_diffusion_kernel, mode="same")
		
		chemical_fields = jnp.concatenate([agents_scent_field[None], chemical_fields],axis=0)
		chemical_fields = jnp.where(chemical_fields<self.cfg.chemicals_detection_threshold, 0.0, chemical_fields) #C,H,W

		agents_chemicals_inputs = jax.vmap(self.vision_fn, in_axes=(None,0))(chemical_fields, agents.position)

		agents_internal_inputs = jnp.concatenate([agents.energy[:,None], agents.reward[:,None]], axis=-1)

		agents_walls_inputs = jax.vmap(self.vision_fn, in_axes=(None,0))(self.walls[None], agents.position)

		return Observation(chemicals=agents_chemicals_inputs, internal=agents_internal_inputs, walls=agents_walls_inputs)

	#-------------------------------------------------------------------

	def _apply_actions(self, state: EnvState, actions: jax.Array)->Tuple[EnvState, dict]:
		"""
		"""
		agents = state.agents_states

		# --- 1. Move ---

		new_positions = jax.vmap(self.agent_interface.move)(actions, agents.position)
		hits_wall = jax.vmap(lambda p: jnp.any(self.walls[*get_cell_index(self.agent_interface.full_body_pos(p))]))(new_positions)
		hits_wall = hits_wall & agents.alive
		if self.cfg.wall_effect=="kill":
			agents_alive = agents.alive&(~hits_wall)
			agents_energy = agents.energy
		elif self.cfg.wall_effect=="penalize":
			agents_alive = agents.alive
			agents_energy = jnp.where(hits_wall&agents.alive, agents.energy-10.0, agents.energy)
		elif self.cfg.wall_effect=="none":
			agents_alive = agents.alive
			agents_energy = agents.energy
		else:
			raise ValueError(f"wall effect {self.cfg.wall_effect} is not valid")

		agents = agents.replace(
			position = new_positions,
			alive    = agents_alive,
			energy   = agents_energy
		)

		# --- 2. Eat ---

		food = state.food
		eating_agents = agents.alive & (agents.energy<self.cfg.max_energy) #can only eat if not full and alive
		body_cells = get_cell_index(jax.vmap(self.agent_interface.full_body_pos)(agents.position)) #N,2,S,S
		*_, S = body_cells.shape
		eating_agents_expanded = jnp.tile(eating_agents[:,None,None], (1,S,S))

		eating_agents_grid = (jnp.zeros(self.cfg.size, dtype=jnp.uint8)
							  .at[*body_cells.transpose(1,0,2,3).reshape(2,-1)]
							  .add(eating_agents_expanded.reshape(-1)))
		energy_grid = jnp.sum(food*self.food_types.energy_concentration[:,None,None], axis=0) #total qty of energy in each cell
		energy_per_agent_grid = jnp.where(eating_agents_grid>0, energy_grid/eating_agents_grid, 0.0) #H,W

		agents_energy_intake = jax.vmap(
			lambda cells: jnp.sum(energy_per_agent_grid[*cells]) 
		)(body_cells)

		agents_energy = jnp.clip(agents.energy + agents_energy_intake, -jnp.inf, self.cfg.max_energy)

		agents = agents.replace(energy=agents_energy)
		food = jnp.where(eating_agents_grid[None]>0, False, food)

		return (
			state.replace(agents_states=agents, food=food), 
			{
				"energy_intakes":agents_energy_intake,
				"dead_by_wall": hits_wall
			}
		)

	#-------------------------------------------------------------------

	@property
	def n_food_types(self):
		return self.food_types.growth_rate.shape[0]

	def _init_food(self, key)->FoodMap:
		food = jr.bernoulli(key, self.food_types.initial_density[:,None,None], (self.n_food_types, *self.cfg.size))
		food = jnp.where(jnp.cumsum(food.astype(jnp.uint4),axis=0)>food, False, food)
		return food

	# ---

	def _update_food(self, state: EnvState, key: jax.Array):
		"""Do one step of food growth"""
		food = state.food
		# --- Grow ---
		p_grow = self.growth_conv(food); assert isinstance(p_grow, jax.Array)
		p_grow = jnp.where(jnp.any(food, axis=0, keepdims=True), 0.0, p_grow)
		grow = jr.bernoulli(key, p_grow)

		grow = jnp.where(
			jnp.cumsum(grow.astype(jnp.uint4),axis=0)>1 | self.walls[None],
			False,
			grow
		)

		food = food | grow

		return state.replace(food=food)

	#-------------------------------------------------------------------

	def render(self, state: EnvState, ax:plt.Axes|None=None):

		if ax is None:
			ax = plt.figure().add_subplot()
		else:
			ax=ax
		assert ax is not None

		food = state.food # F, X, Y
		F, H, W = food.shape
		agents = state.agents_states
		food_colors = plt.cm.Set2(jnp.arange(food.shape[0])) #type:ignore

		img = jnp.ones((F,H,W,4)) * food_colors[:,None,None]
		img = jnp.clip(jnp.where(food[...,None], img, 0.).sum(0), 0.0, 1.0) #type:ignore
		img = img.at[:,:,-1].set(jnp.any(food, axis=0))

		img = jnp.where(self.walls[...,None], jnp.array([0.5, 0.5, 0.5, 1.0]), img)

		ai, aj = agents.position.pos[agents.alive].T
		ax.scatter(aj,ai, marker="s", color="k")
		ax.imshow(img, origin="lower")

	# ---

	def render_states(self, states: list|EnvState, ax: plt.Axes, cam: Camera):

		if isinstance(states, EnvState):
			T = states.time.shape[0]
			states = [jax.tree.map(lambda x:x[t], states) for t in range(T)]

		for state in states:
			self.render(state, ax)
			cam.snap()

		return cam


if __name__ == '__main__':
	pass

