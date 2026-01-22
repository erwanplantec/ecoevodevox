from .simulation import Simulator
from .core import SimulationState

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.axes import Axes
import jax, jax.numpy as jnp, jax.random as jr, jax.nn as jnn

def render(simulator: Simulator, sim_state: SimulationState, ax:Axes|None=None):

    if ax is None:
        ax = plt.figure().add_subplot()
    else:
        ax=ax
    assert ax is not None

    food = sim_state.env_state.food # F, X, Y
    F, H, W = food.shape
    agents = sim_state.agents_states
    food_colors = plt.cm.Set2(jnp.arange(food.shape[0])) #type:ignore

    img = jnp.ones((F,H,W,4)) * food_colors[:,None,None]
    img = jnp.clip(jnp.where(food[...,None], img, 0.).sum(0), 0.0, 1.0) #type:ignore
    img = img.at[:,:,-1].set(jnp.any(food, axis=0))

    img = jnp.where(sim_state.env_state.walls[...,None], jnp.array([0.5, 0.5, 0.5, 1.0]), img)

    colormap = lambda e: plt.cm.winter((e / (simulator.agent_interface.cfg.max_energy*2) + 1) /2) #type:ignore
    for a in range(agents.alive.shape[0]):
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
