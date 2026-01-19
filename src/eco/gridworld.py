from .chemicals import ChemicalType, make_chemical_diffusion_convolution
from .food import FoodType, FoodMap,  make_growth_convolution

from flax.struct import PyTreeNode
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import equinox as eqx
from celluloid import Camera

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

from typing import Callable, Literal, NamedTuple, Tuple
from jaxtyping import (
	Float, PyTree, Array,
	Bool,
	UInt32,
	Float16, Float32
)
from ..agents.interface import AgentInterface, AgentState, Genotype, Body

# ======================== UTILS =============================

type KeyArray = jax.Array
type AgentParams = jax.Array
type Action = jax.Array
type Info = dict

@jax.jit
def get_cell_index(pos: Float16):
	indices = jnp.floor(pos).astype(jnp.int16)
	return indices

# ============================================================

class EnvState(PyTreeNode):
	agents_states: AgentState # state of agents
	food: FoodMap # 2d map indicating food locations
	time: UInt32
	last_agent_id: UInt32=0

class Observation(PyTreeNode):
	chemicals: jax.Array #C,H,W
	internal: jax.Array
	walls: jax.Array

#=======================================================================

class GridworldConfig(PyTreeNode):
	# ---
	size: tuple[int,int]=(256,256)  # Grid dimensions (height, width)
	# ---
	walls_density: float=1e-4  							   # Probability of wall placement at each grid cell
	wall_effect: Literal["kill","penalize","none"]="kill"  # What happens when agents hit walls
	wall_penalty: float=1.0  							   # Penalty for hitting walls (if wall_effect==penalize)
	# ---
	max_agents: int=10_000  # Maximum number of agents that can exist simultaneously
	init_agents: int=1_024  # Initial number of agents at environment start
	max_age: int=1_000  	# Maximum age before agents die of old age
	# ---
	reproduction_cost: float=0.5  				# Energy cost for reproducing
	max_energy: float=50.0  					# Maximum energy an agent can have
	initial_energy: float=1.0  					# Starting energy for new agents
	time_above_threshold_to_reproduce: int=100  # Time steps above energy threshold needed to reproduce
	time_below_threshold_to_die: int=30  		# Time steps below energy threshold before death
	# ---
	chemicals_detection_threshold: float=1e-3  # Minimum chemical concentration for detection
	# ---
	birth_pool_size: int=256  # Size of pool for managing births (sets maximum births per step)
	# ---
	agent_scent_diffusion_rate: float=0.1
	flow: jax.Array|tuple[float,float]|None=None  # Environmental flow field affecting chemical diffusion

#=======================================================================

class GridWorld:
	
	#-------------------------------------------------------------------

	def __init__(
		self, 
		cfg: GridworldConfig,
		agent_interface: AgentInterface,
		mutation_fn: Callable[[Genotype,jax.Array], Genotype],
		chemical_types: ChemicalType,
		food_types: FoodType, 
		*,
		key: jax.Array):

		# ---
		nb_food_types = food_types.growth_rate.shape[0]
		nb_chemical_types = chemical_types.diffusion_rate.shape[0]
		assert food_types.chemical_signature.shape==(nb_food_types, nb_chemical_types)
		# ---
		
		self.cfg = cfg
		self.walls = jr.bernoulli(key, cfg.walls_density, cfg.size)
		self.food_types = jax.tree.map(lambda x: x.astype(jnp.float16), food_types)
		self.nb_food_types = food_types.growth_rate.shape[0]
		self.growth_conv = make_growth_convolution(cfg.size, 
												   food_types.growth_rate,
												   food_types.dmin, 
												   food_types.dmax)
	
		self.chemical_types = chemical_types
		flow = cfg.flow
		if cfg.flow is not None:
			flow = jnp.asarray(cfg.flow)
			flow = None if jnp.allclose(flow,0.0) else flow

		self.chemicals_diffusion_conv = make_chemical_diffusion_convolution(cfg.size,
																			chemical_types.diffusion_rate,
																			flow=flow) #type:ignore

		@jax.jit
		def _vision_fn(x: jax.Array, body: Body):
			"""Return sample of x at body discretization points"""
			indices = get_cell_index(agent_interface.get_body_points(body))
			return x[:, *indices] if x.ndim==3 else x[*indices]

		self.vision_fn = _vision_fn

		self.agent_interface = agent_interface
		self.mutation_fn = mutation_fn
		self.agent_scent_diffusion_conv = make_chemical_diffusion_convolution(cfg.size, 
	                                                                          jnp.array([cfg.agent_scent_diffusion_rate]), 
	                                                                          flow=flow) #type:ignore

	#-------------------------------------------------------------------

	def step(self, state: EnvState, key: jax.Array)->Tuple[EnvState,PyTree]:

		# --- 1. Update food sources ---
		key, key_food = jr.split(key)
		state = self._update_food(state, key_food)

		# --- 2. Get and apply actions ---
		key, key_obs, key_step = jr.split(key, 3)
		observations = self._get_observations(state, key_obs)
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
			dict(actions=actions, 
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

	# ---

	def _init_food(self, key: jax.Array)->FoodMap:
		food = jr.bernoulli(key, self.food_types.initial_density[:,None,None], (self.n_food_types, *self.cfg.size))
		food = jnp.where(jnp.cumsum(food.astype(jnp.uint4),axis=0)>1, False, food)
		return food

	#-------------------------------------------------------------------

	def _init_agents(self, key)->AgentState:

		def _pad(x):
			pad_values = [(0,self.cfg.max_agents-self.cfg.init_agents)] + [(0,0)]*(x.ndim-1)
			return jnp.pad(x, pad_values)

		key_prms, key_pos, key_head, key_size, key_init = jr.split(key, 5)
		alive = jnp.ones(self.cfg.init_agents, dtype=bool)
		policy_params = jax.vmap(self.agent_interface.policy_fctry)(jr.split(key_prms,self.cfg.init_agents))
		body_sizes = jr.uniform(key_size, (self.cfg.init_agents,), minval=self.agent_interface.min_body_size, maxval=self.agent_interface.max_body_size, dtype=jnp.float16) 
		genotypes = Genotype(policy_params, body_sizes)
		policy_states, sensory_states, motor_states, body_sizes = jax.vmap(self.agent_interface.init)(
			genotypes, jr.split(key_init, self.cfg.init_agents)
		)
		positions = jr.uniform(key_pos, (self.cfg.init_agents, 2), minval=1.0, maxval=jnp.array(self.cfg.size, dtype=jnp.float16)-1, dtype=jnp.float16)
		headings = jr.uniform(key_head, (self.cfg.init_agents,), minval=0.0, maxval=2*jnp.pi, dtype=jnp.float16)
		bodies = Body(positions, headings, body_sizes)

		states = AgentState(
			genotype             = genotypes,
			# ---, 
			policy_state 		 = policy_states,
			sensory_state 		 = sensory_states,
			motor_state 		 = motor_states,
			# ---
			body 			     = bodies, 
			# ---
			alive 				 = alive, 
			energy 				 = jnp.full((self.cfg.init_agents), self.cfg.initial_energy, dtype=jnp.float16)*alive, 
			time_above_threshold = jnp.full((self.cfg.init_agents,), 0, dtype=jnp.uint16), 
			time_below_threshold = jnp.full((self.cfg.init_agents,), 0, dtype=jnp.uint16),
			# ---
			reproduce 			 = jnp.full((self.cfg.init_agents,), False, dtype=bool),
			reward 				 = jnp.zeros((self.cfg.init_agents,), dtype=jnp.float16), 
			# ---
			age 				 = jnp.ones((self.cfg.init_agents), dtype=jnp.uint16), 
			n_offsprings 		 = jnp.zeros(self.cfg.init_agents, dtype=jnp.uint16),
			id_ 				 = jnp.where(alive, jnp.cumsum(alive, dtype=jnp.uint32), 0),
			parent_id_ 			 = jnp.zeros(self.cfg.init_agents, dtype=jnp.uint32),
			generation 			 = jnp.zeros(self.cfg.init_agents, dtype=jnp.uint16),
		)

		states = jax.tree.map(_pad, states)

		return states

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
			parents_genotypes = jax.tree.map(lambda x: x[parents_buffer_id], agents.genotype)
			is_free, childs_buffer_id = jax.lax.top_k(free_buffer_spots, self.cfg.birth_pool_size)
			childs_mask = parents_mask & is_free #is a child if parent was actually reproducing and there are free buffer spots
			childs_buffer_id = jnp.where(childs_mask, childs_buffer_id, self.cfg.max_agents) # assign wrong index if not born
			parents_buffer_id = jnp.where(childs_mask, parents_buffer_id, self.cfg.max_agents)

			childs_alive = childs_mask
			
			childs_genotypes = jax.vmap(self.mutation_fn)(parents_genotypes, jr.split(key_mut, self.cfg.birth_pool_size))
			childs_policy_states, childs_sensory_states, childs_motor_states, childs_sizes = jax.vmap(self.agent_interface.init)(
				childs_genotypes, jr.split(key_init, self.cfg.birth_pool_size)
			)
			
			childs_energy = jnp.full(self.cfg.birth_pool_size, self.cfg.initial_energy, dtype=jnp.float16)

			parents_bodies = jax.tree.map(lambda x: x[parents_buffer_id], agents.body)
			direction = jnp.mod(parents_bodies.heading + jnp.pi, 2*jnp.pi)
			delta = jnp.stack([jnp.cos(direction), jnp.sin(direction)], axis=-1)
			childs_positions = agents.body.pos[parents_buffer_id] + delta*(parents_bodies.size+childs_sizes+0.1)[:,None] 
			childs_headings = jr.uniform(key_head, minval=0.0, maxval=2*jnp.pi, dtype=jnp.float16)
			childs_bodies = Body(childs_positions, childs_headings, childs_sizes)

			agents_alive = agents.alive.at[childs_buffer_id].set(childs_alive) #make sur to not overwrite occupied buffer ids (if more reproducers than free buffer spots)
			
			agents_genotypes = jax.tree.map(
				lambda x, x_child: x.at[childs_buffer_id].set(x_child),
				agents.genotype, childs_genotypes
			)
			
			agents_policy_states = jax.tree.map(lambda x, c: x.at[childs_buffer_id].set(c), agents.policy_state, childs_policy_states)
			agents_sensory_states = jax.tree.map(lambda x, c: x.at[childs_buffer_id].set(c), agents.sensory_state, childs_sensory_states)
			agents_motor_states = jax.tree.map(lambda x, c: x.at[childs_buffer_id].set(c), agents.motor_state, childs_motor_states)

			agents_energy = agents.energy.at[childs_buffer_id].set(childs_energy)
			agents_energy = agents_energy.at[parents_buffer_id].add(-self.cfg.reproduction_cost * childs_mask)
			
			agents_bodies = jax.tree.map(lambda x, c: x.at[childs_buffer_id].set(c), agents.body, childs_bodies)
			
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

			childs_ids = jnp.where(childs_mask, jnp.cumsum(childs_mask, dtype=jnp.uint32)+state.last_agent_id+1, 0)
			agents_id = agents.id_.at[childs_buffer_id].set(childs_ids)

			childs_parent_id = agents.id_[parents_buffer_id]
			agents_parent_ids = agents.parent_id_.at[childs_buffer_id].set(childs_parent_id)

			parents_generation = agents.generation[parents_buffer_id]
			agents_generation = agents.generation.at[childs_buffer_id].set(parents_generation+1)

			agents = AgentState(
				genotype		 	 = agents_genotypes,
				# ----
				policy_state 		 = agents_policy_states,
				sensory_state 		 = agents_sensory_states,
				motor_state 		 = agents_motor_states,
				# ---
				body 			 	 = agents_bodies,
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

		reproducing = agents.alive \
				    & (agents.time_above_threshold > self.cfg.time_above_threshold_to_reproduce) \
				    & jnp.any(~agents.alive)


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

	def _compute_chemical_fields(self, state: EnvState, key: jax.Array)->jax.Array:

		chemical_source_fields = jnp.sum(state.food[:,None] * self.food_types.chemical_signature[...,None,None], axis=0)
		chemical_fields = self.chemicals_diffusion_conv(chemical_source_fields) #C,H,W
		chemical_fields = jnp.where(
			self.chemical_types.is_sparse[:,None,None], 
			jr.bernoulli(key, p=chemical_fields*self.chemical_types.emission_rate[...,None,None]).astype(jnp.float16), 
			chemical_fields ); assert isinstance(chemical_fields, jax.Array)
		agents = state.agents_states
		agents_i, agents_j = get_cell_index(agents.body.pos).T
		agents_alive_grid = jnp.zeros(self.cfg.size).at[agents_i, agents_j].add(agents.alive)
		agents_scent_field = self.agent_scent_diffusion_conv(agents_alive_grid[None])
		
		chemical_fields = jnp.concatenate([agents_scent_field, chemical_fields],axis=0)
		chemical_fields = jnp.where(chemical_fields<self.cfg.chemicals_detection_threshold, 0.0, chemical_fields) #C,H,W

		return chemical_fields

	#-------------------------------------------------------------------

	def _get_observations(self, state: EnvState, key: jax.Array)->Observation:
		"""
		returns agents observations
		"""
		agents = state.agents_states

		chemical_fields = self._compute_chemical_fields(state, key)

		agents_chemicals_inputs = jax.vmap(self.vision_fn, in_axes=(None,0))(chemical_fields, agents.body)

		agents_internal_inputs = jnp.concatenate([agents.energy[:,None], agents.reward[:,None]], axis=-1)

		agents_walls_inputs = jax.vmap(self.vision_fn, in_axes=(None,0))(self.walls[None], agents.body)

		return Observation(chemicals=agents_chemicals_inputs, internal=agents_internal_inputs, walls=agents_walls_inputs)

	#-------------------------------------------------------------------

	def _normalize_position(self, body: Body):
		pos = jnp.mod(body.pos, jnp.array(self.cfg.size))
		heading = jnp.mod(body.heading, 2*jnp.pi)
		return body.replace(pos=pos, heading=heading)

	def _apply_actions(self, state: EnvState, actions: jax.Array)->Tuple[EnvState, dict]:
		"""
		"""
		agents = state.agents_states

		# --- 1. Move ---

		new_positions = jax.vmap(self.agent_interface.move)(actions, agents.body)
		new_positions = self._normalize_position(new_positions)
		hits_wall = jax.vmap(lambda p: jnp.any(self.walls[*get_cell_index(self.agent_interface.get_body_points(p))]))(new_positions)
		hits_wall = hits_wall & agents.alive
		if self.cfg.wall_effect=="kill":
			agents_alive = agents.alive&(~hits_wall)
			agents_energy = agents.energy
		elif self.cfg.wall_effect=="penalize":
			agents_alive = agents.alive
			agents_energy = jnp.where(hits_wall&agents.alive, agents.energy-self.cfg.wall_penalty, agents.energy)
		elif self.cfg.wall_effect=="none":
			agents_alive = agents.alive
			agents_energy = agents.energy
		else:
			raise ValueError(f"wall effect {self.cfg.wall_effect} is not valid")

		agents = agents.replace(
			body   = new_positions,
			alive  = agents_alive,
			energy = agents_energy
		)

		# --- 2. Eat ---

		food = state.food
		eating_agents = agents.alive & (agents.energy<self.cfg.max_energy) #can only eat if not full and alive
		body_cells = get_cell_index(jax.vmap(self.agent_interface.get_body_points)(agents.body)) #N,2,S,S
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

	def _update_food(self, state: EnvState, key: jax.Array):
		"""Do one step of food growth"""
		food = state.food
		# --- Grow ---
		occupied = jnp.any(food, axis=0)
		p_grow = self.growth_conv(food); assert isinstance(p_grow, jax.Array)
		p_grow = p_grow + self.food_types.spontaneous_grow_prob[:,None,None]
		p_grow = jnp.clip(p_grow, 0.0, 1.0)
		grow = jr.bernoulli(key, p_grow)

		grow = jnp.where(
			(jnp.cumsum(grow,axis=0)>1) | self.walls[None] | occupied[None],
			False,
			grow
		)

		food = food | grow

		return state.replace(food=food)

	#-------------------------------------------------------------------

	def render(self, state: EnvState, ax:Axes|None=None):

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

		colormap = lambda e: plt.cm.winter((e / (self.cfg.max_energy*2) + 1) /2) #type:ignore
		for a in range(self.cfg.max_agents):
			if not agents.alive[a] : continue
			body = jax.tree.map(lambda x: x[a], agents.body)
			x,y = body.pos
			h = body.heading
			e = agents.energy[a]
			s = body.size
			body = Rectangle((x-s/2,y-s/2), s, s, angle=(h/(2*jnp.pi))*360, 
                     facecolor=colormap(e), rotation_point="center")
			ax.add_patch(body)
			dy, dx = jnp.sin(h), jnp.cos(h)
			ax.arrow(x, y, dx*s/2, dy*s/2)

		ax.imshow(img.transpose(1,0,2), origin="lower")

	# ---

	def render_states(self, states: list|EnvState, ax: Axes, cam: Camera):

		if isinstance(states, EnvState):
			T = states.time.shape[0]
			states = [jax.tree.map(lambda x:x[t], states) for t in range(T)]

		for state in states:
			self.render(state, ax)
			cam.snap()

		return cam