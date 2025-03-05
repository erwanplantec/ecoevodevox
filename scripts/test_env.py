from functools import partial
from jax.flatten_util import ravel_pytree
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

import matplotlib.pyplot as plt
from celluloid import Camera

from src.eco.env import GridWorld, FoodType
from src.devo.baselines.mlp import MLPPolicy
from src.devo.utils import make_apply_init
from src.evo.mutation import mutate_flat_generalized


# --- Env ---
env_size = (128,128)
max_agents = 10_000
apply_mb = None
birth_pool_size = max_agents

# --- Food ---
food_types = FoodType(
	reproduction_rate=jnp.array([0.01]),
	diffusion_rate=jnp.array([1.0]),
	energy_concentration=jnp.array([1.0])
)

# --- Agents ---
mlp_depth = 1
mlp_width = 64

agent_model = MLPPolicy(29, width=mlp_width, depth=mlp_depth, key=jr.key(1))
agent_apply, agent_init = make_apply_init(agent_model)
agent_fctry = lambda key: ravel_pytree(eqx.filter(MLPPolicy(29, width=mlp_width, depth=mlp_depth, key=key), eqx.is_array))[0]

# --- Evolution ---
sigma_mut = 0.03
p_mut = 0.01
mutation_fn = partial(mutate_flat_generalized, sigma=sigma_mut, p=p_mut)

# --- Instantiate ---

env = GridWorld(agent_fctry, agent_init, agent_apply, food_types, env_size, mutation_fn, #type:ignore
	max_agents=max_agents, init_agents=200, max_age=200, reproduction_energy_cost=2, initial_food_density=0.01, 
	time_above_threshold_to_reproduce=50, time_below_threshold_to_die=30,
	base_energy_loss=0.1) 

def step_fn(state, key):
	state, data = env.step(state, key)
	data = eqx.tree_at(lambda d: d["state"].agents.prms, data, None)
	return state, data

# --- Simulate ---

state = env.reset(jr.key(1))

state, data = jax.lax.scan(step_fn, state, jr.split(jr.key(2), 1024))

states = data["state"]

T = states.time.shape[0]


fig, ax = plt.subplots(4, 1, figsize=(16,16), height_ratios=[15,3,3,3])

cam = Camera(fig)
for t in range(T):
	state = jax.tree.map(lambda x: x[t], states)
	env.render(state, ax[0])
	ax[1].plot(states.agents.alive[:t].sum(-1), color="k")
	ax[2].plot(states.agents.energy[:t].mean(-1), color="k")
	ax[2].plot(states.agents.energy[:t,0], color="k")
	ax[3].imshow(jnp.stack([state.agents.alive]*10))
	cam.snap()
ani = cam.animate(interval=100)
ani.save("anims/animation.mp4")

