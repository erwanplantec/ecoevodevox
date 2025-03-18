import jax
import jax.numpy as jnp
import jax.random as jr

from src.eco.gridworld import FoodType, GridWorld

# ======= Simulation Parameters =========

passive_eating = True
passive_reproduction = False
predation = False
move_energy_cost = 0.1
base_energy_cost = 0.1
time_to_reproduce = 50
time_to_die = 20


# ============= Ediacarian ==============

ediacarian_start_time = 0
n_ediacarian_food_types = 1

ediacarian_food_types = FoodType(
	reproduction_rate=jnp.array([0.1]*n_ediacarian_food_types),
	expansion_rate=jnp.array([1.]*n_ediacarian_food_types),
	max_concentration=jnp.array([1]*n_ediacarian_food_types, dtype=jnp.int16),
	chemical_signature=jnp.identity(n_ediacarian_food_types),
	energy_concentration=jnp.array([1.]*n_ediacarian_food_types),
	spontaneous_grow_prob=jnp.array([0.0]*n_ediacarian_food_types)
)


# ============== Cambrian ===============

cambrian_start_time = 200

n_cambrian_food_types = 7
cambrian_food_types = FoodType(
	reproduction_rate=jnp.array([0.1]*n_cambrian_food_types),
	expansion_rate=jnp.array([0.1]*n_cambrian_food_types),
	max_concentration=jnp.array([10]*n_cambrian_food_types, dtype=jnp.int16),
	chemical_signature=jnp.identity(n_cambrian_food_types),
	energy_concentration=jnp.array([1.]*n_cambrian_food_types),
	spontaneous_grow_prob=jnp.array([0.0]*n_cambrian_food_types)
)

# =============== Agents =================

init_agents = 100
max_agents = 10_000
field_of_view = 1

agents = ...

# ============== Simulation Setup ============

n_food_types = n_ediacarian_food_types + n_cambrian_food_types

ediacarian_env = ...
cambrian_env = 

# ============== Simulation ==================








