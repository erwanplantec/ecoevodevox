from functools import partial
from jax.flatten_util import ravel_pytree
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import equinox as eqx

import matplotlib.pyplot as plt
from celluloid import Camera

from src.eco.gridworld import ChemicalType, GridWorld, FoodType
from src.devo.baselines.mlp import MLPPolicy
from src.devo.utils import make_apply_init
from src.evo.mutation import mutate_flat_generalized


# --- Env ---
env_size = (32, 64)
max_agents = 1
apply_mb = None
birth_pool_size = max_agents
init_agents = 0
move_cost = 0.05
reproduction_cost = 0.1
base_cost = 0.1

# --- Food ---
n_food = n_chemicals = 1
food_types = FoodType(
	growth_rate=jnp.array([0.2]*n_food, dtype=jnp.float16),
	dmin=jnp.array([4.]),
	dmax=jnp.array([8.]),
	energy_concentration=jnp.array([1.0]*n_food, dtype=jnp.float16),
	chemical_signature=jnp.identity(n_chemicals, dtype=jnp.float16),
	spontaneous_grow_prob=jnp.zeros(n_food, dtype=jnp.float16),
	initial_density=jnp.array([0.001])
)
chemical_types = ChemicalType(diffusion_rate=jnp.array([1.0]*n_chemicals))
initial_food_density = 0.005 / n_food

# --- Agents ---
mlp_depth = 0
mlp_width = 16
n_inputs = (n_chemicals+2)*9 + 2
agent_model = MLPPolicy(n_inputs, 5, width=mlp_width, depth=mlp_depth, key=jr.key(1))
agent_apply, agent_init = make_apply_init(agent_model)
agent_fctry = lambda key: ravel_pytree(eqx.filter(MLPPolicy(n_inputs, 5, width=mlp_width, depth=mlp_depth, key=key), eqx.is_array))[0]

# --- Evolution ---
sigma_mut = 0.01
p_mut = 1.0
mutation_fn = partial(mutate_flat_generalized, sigma=sigma_mut, p=p_mut)

# --- Instantiate --- 

env = GridWorld(env_size, agent_fctry, agent_init, agent_apply, mutation_fn, chemical_types, food_types,  #type:ignore
	max_agents=max_agents, init_agents=init_agents, max_age=1_000, reproduction_energy_cost=reproduction_cost, 
	move_energy_cost=move_cost, base_energy_loss=base_cost, 
	initial_agent_energy=1.0,
	time_above_threshold_to_reproduce=50, time_below_threshold_to_die=30,
	passive_reproduction=True, passive_eating=True, predation=False,) 

def step_fn(state, key):
	state, data = env.step(state, key)
	data = eqx.tree_at(lambda d: d["state"].agents.prms, data, None)
	return state, data

# --- Simulate ---

state = env.reset(jr.key(0))


state, data = jax.lax.scan(step_fn, state, jr.split(jr.key(3), 100))

states = data["state"]

T = states.time.shape[0]

fig, ax = plt.subplots(2, 1, figsize=(16,16))

cam = Camera(fig) #type:ignore
energy_content = (states.food * food_types.energy_concentration[None,:,None,None]).sum((2,3))
for t in range(T):
	state = jax.tree.map(lambda x: x[t], states)
	env.render(state, ax[0])
	chemical_fields = jnp.sum(state.food[:,None] * env.food_types.chemical_signature[...,None,None], axis=0)
	chemical_fields = env.chemicals_diffusion_conv(chemical_fields)
	cmax = chemical_fields.max()
	for i, C in enumerate(chemical_fields):
		ax[1].imshow(C[...,None] * jnp.array(plt.cm.Set2(i))[None,None] / cmax) #type:ignore
	#ax[1].plot(energy_content[:t].sum(1), color=plt.cm.Set2(0))
	cam.snap()
ani = cam.animate(interval=100)
ani.save("anims/animation.mp4")

# fig, ax = plt.subplots(3, 1, figsize=(8,4))

# ax[0].plot(states.agents.alive.sum(-1), color="k")
# ax[1].plot(states.agents.energy.mean(-1), color="k")
# ax[1].plot(states.agents.energy, color="k")
# ax[2].plot(states.food.sum((1,2,3)))
# plt.show()

