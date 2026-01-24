from .chemicals import ChemicalType, make_chemical_diffusion_convolution
from .food import FoodType, FoodMap,  make_growth_convolution

from flax.struct import PyTreeNode
import jax, jax.numpy as jnp, jax.random as jr, jax.nn as jnn
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
from ..devo.core import Body

# ======================== UTILS =============================

type KeyArray = jax.Array
type AgentParams = jax.Array
type Action = jax.Array
type Info = dict
type WallsMap = Bool[jax.Array, "H W"]

@jax.jit
def get_cell_index(pos: Float16):
	indices = jnp.floor(pos).astype(jnp.int16)
	return indices

# ============================================================

class EnvState(PyTreeNode):
	food: FoodMap # 2d map indicating food locations
	walls: WallsMap
	time: UInt32
	last_agent_id: UInt32=0


type Observation = Float[jax.Array, "C+1 R R"] # observation has C(nb chems) + 1(walls) channels, R is body resolution

#=======================================================================

class GridworldConfig(PyTreeNode):
	# ---
	size: tuple[int,int]=(256,256)  # Grid dimensions (height, width)
	# ---
	walls_density: float=1e-4  		# Probability of wall placement at each grid cell
	# ---
	chemicals_detection_threshold: float=1e-3  # Minimum chemical concentration for detection
	# ---
	flow: jax.Array|tuple[float,float]|None=None  # Environmental flow field affecting chemical diffusion

#=======================================================================

class GridWorld:
	
	#-------------------------------------------------------------------

	def __init__(
		self, 
		cfg: GridworldConfig,
		chemical_types: ChemicalType,
		food_types: FoodType):

		# ---
		nb_food_types = food_types.growth_rate.shape[0]
		nb_chemical_types = chemical_types.diffusion_rate.shape[0]
		assert food_types.chemical_signature.shape==(nb_food_types, nb_chemical_types)
		# ---
		
		self.cfg = cfg
		self.food_types: FoodType = jax.tree.map(lambda x: x.astype(jnp.float16), food_types)
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
		def _vision_fn(x: jax.Array, body_points: jax.Array)->jax.Array:
			"""Return sample of x at body discretization points"""
			indices = get_cell_index(body_points)
			return x[:, *indices] if x.ndim==3 else x[*indices]

		self.vision_fn = _vision_fn
	
	#-------------------------------------------------------------------

	@property
	def nb_chemicals(self):
		return self.chemical_types.diffusion_rate.shape[0]

	@property
	def nb_food_types(self):
		return self.food_types.growth_rate.shape[0]

	#-------------------------------------------------------------------

	def init(self, *, key: jax.Array)->EnvState:
		key_food, key_walls = jr.split(key)
		food_sources = self._init_food(key=key_food)
		walls = jr.bernoulli(key_walls, self.cfg.walls_density, self.cfg.size)
		return EnvState(food=food_sources, time=jnp.zeros((), dtype=jnp.uint32), walls=walls)

	#-------------------------------------------------------------------

	def _init_food(self, *, key: jax.Array)->FoodMap:
		food = jr.bernoulli(key, self.food_types.initial_density[:,None,None], (self.nb_food_types, *self.cfg.size))
		food = jnp.where(jnp.cumsum(food.astype(jnp.uint4),axis=0)>1, False, food)
		return food

	#-------------------------------------------------------------------

	def compute_chemical_fields(self, state: EnvState, other_sources: jax.Array|None=None, *, key: jax.Array)->jax.Array:
		"""Computes chemical field emitted by food sources  and ither externally provided sources"""
		
		food_chemical_source_fields = jnp.sum(state.food[:,None] * self.food_types.chemical_signature[...,None,None], axis=0)
		if other_sources is not None:
			assert other_sources.shape == food_chemical_source_fields.shape, f"other_sources has the wrong shape. Expected {food_chemical_source_fields.shape}, got {other_sources.shape}"
			chemical_source_fields = food_chemical_source_fields + other_sources
		else:
			chemical_source_fields = food_chemical_source_fields
		
		chemical_fields = self.chemicals_diffusion_conv(chemical_source_fields) #C,H,W
		chemical_fields = jnp.where(
			self.chemical_types.is_sparse[:,None,None], 
			jr.bernoulli(key, p=chemical_fields*self.chemical_types.emission_rate[...,None,None]).astype(jnp.float16), 
			chemical_fields ); assert isinstance(chemical_fields, jax.Array)

		chemical_fields = jnp.where(chemical_fields<self.cfg.chemicals_detection_threshold, 0.0, chemical_fields)
		
		return chemical_fields

	#-------------------------------------------------------------------

	def get_agents_observations(self, state: EnvState, bodies_points: jax.Array, other_chemical_sources: jax.Array|None=None, *, key: jax.Array)->Observation:
		"""
		returns agents observations
		"""
		chemical_fields = self.compute_chemical_fields(state, other_chemical_sources, key=key)

		agents_chemicals_inputs = jax.vmap(self.vision_fn, in_axes=(None,0))(chemical_fields, bodies_points)

		agents_walls_inputs = jax.vmap(self.vision_fn, in_axes=(None,0))(state.walls[None], bodies_points)

		return jnp.concatenate([agents_chemicals_inputs, agents_walls_inputs], axis=1)

	#-------------------------------------------------------------------

	def normalize_posture(self, body: Body):
		"""keep position within bounds and heading within [0, 2*pi]"""
		pos = jnp.mod(body.pos, jnp.array(self.cfg.size))
		heading = jnp.mod(body.heading, 2*jnp.pi)
		return body.replace(pos=pos, heading=heading)

	#-------------------------------------------------------------------

	def check_wall_contact(self, env_state: EnvState, body_points: jax.Array):
		"""check if body contacts with walls"""
		makes_contact = jnp.any(env_state.walls[*get_cell_index(body_points)])
		return makes_contact

	#-------------------------------------------------------------------

	def share_food_and_update(self, env_state: EnvState, bodies_points: jax.Array, agents_mask: jax.Array)->tuple[jax.Array, EnvState]:
		"""computes how much energy goes to each agent and update the food map accordingly"""

		bodies_cells = get_cell_index(bodies_points)
		*_, body_resolution = bodies_cells.shape

		agents_mask_exp = jnp.tile(agents_mask[:,None,None], (1,body_resolution,body_resolution))

		# computes the number of agents body parts in each cell of the env
		nb_body_parts_per_cell = (jnp.zeros(self.cfg.size, dtype=jnp.uint8)
							  	  .at[*bodies_cells.transpose(1,0,2,3).reshape(2,-1)]
							  	  .add(agents_mask_exp.reshape(-1)))

		# computes the available amount of energy in each cell
		energy_grid = jnp.sum(env_state.food*self.food_types.energy_concentration[:,None,None], axis=0)

		# computes the quantity of energy per agent in each cell
		energy_per_body_part_per_cell = jnp.where(nb_body_parts_per_cell>0, energy_grid/nb_body_parts_per_cell, 0.0)

		# computes each agents total energy intake
		agents_energy_intake = jax.vmap(
			lambda cells: jnp.sum(energy_per_body_part_per_cell[*cells])
		)(bodies_cells)

		# update food map according to agents eating it
		new_food_map = jnp.where(nb_body_parts_per_cell[None]>0, False, env_state.food)

		return agents_energy_intake, env_state.replace(food=new_food_map)

	#-------------------------------------------------------------------

	def update_food(self, state: EnvState, key: jax.Array):
		"""Do one step of food growth"""
		food = state.food
		# --- Grow ---
		occupied = jnp.any(food, axis=0)
		p_grow = self.growth_conv(food); assert isinstance(p_grow, jax.Array)
		p_grow = p_grow + self.food_types.spontaneous_grow_prob[:,None,None]
		p_grow = jnp.clip(p_grow, 0.0, 1.0)
		grow = jr.bernoulli(key, p_grow)

		grow = jnp.where(
			(jnp.cumsum(grow,axis=0)>1) | state.walls[None] | occupied[None],
			False,
			grow
		)

		food = food | grow

		return state.replace(food=food)

	#-------------------------------------------------------------------