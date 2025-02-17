import jax
from typing import Callable, NamedTuple, Tuple
from jaxtyping import Float, Int, PyTree

class EnvConfig(NamedTuple):
	n_food_types: int
	noise: float

class Agent(NamedTuple):
	# ---
	state: PyTree
	# ---
	position: jax.Array
	heading: jax.Array
	energy: Float
	time_above_threshold: Int
	time_below_threshold: Int
	# ---
	model: PyTree

class FoodType(NamedTuple):
	growth_rate: Float
	diffusion_rate: Float
	chemical_identity: Float

class FoodSource(NamedTuple):
	position: jax.Array
	params: FoodType

class EnvState(NamedTuple):
	agents: Agent
	food: FoodSource

class Environment:
	# ---
	def __init__(self, agents: Agent, food_types: FoodType, size: Tuple[int, int], mutation_fn: Callable):
		
		self.size = size
		self.food_types = food_types
		self.mutation_fn = mutation_fn

	# ---

	def step(self, state: EnvState, key: jax.Array)->EnvState:
		# --- 1. Get and apply actions ---

		# --- 2. Update food sources ---

		# --- 3. Update agents energy ---

		# --- 4. Die / reproduce ---

		return state

	# ---

	def reset(self, key: jax.Array)->EnvState:
		pass

	# ---

	def _compute_agent_inputs(self, state: EnvState)->jax.Array:
		pass





