from .simulation import Simulator
from .core import SimulationState

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.axes import Axes
import numpy as np
import jax, jax.numpy as jnp, jax.random as jr, jax.nn as jnn


def render_frame(simulator: Simulator, sim_state: SimulationState, color_by: str="energy",
                 agent_px: int=1, downsample: int=1,
                 background: tuple[int,int,int]=(12, 12, 20),
                 overlay: np.ndarray|None=None, overlay_gamma: float=0.5,
                 overlay_cmap: str="magma") -> np.ndarray:
	"""Render the world to an RGB uint8 image, fully vectorised.

	Unlike `render()` (which adds one matplotlib patch per agent and takes seconds for a large
	population), this scatters all agents at once with numpy indexing, so it is fast enough to
	drive a live viewer.

	Args:
		simulator: used for the agent energy scale.
		sim_state: state to draw.
		color_by: "energy" (blue->green by energy), "speed" (distance/age), "age" (by age,
			normalised to max_age), "nb_neurons" (grown-network size, scaled to the population
			max) or "flat".
		agent_px: half-width in pixels of the square drawn per agent; >0 keeps agents visible
			when the image is scaled down for display.
		downsample: stride applied at the end for very large worlds (1 = full resolution).
		background: RGB of empty cells.
		overlay: optional `[X, Y]` scalar field (e.g. one chemical channel from
			`Simulator.chemical_fields`) blended under the agents. Drawn beneath them so it never
			hides the population. Normalised by its own max, so it shows *structure*, not
			absolute concentration — the colour scale is not comparable across frames.
		overlay_gamma: exponent applied to the normalised field. <1 lifts faint values, which
			matters because diffused chemicals are dominated by a few bright source cells and a
			linear scale renders the informative tail as black.
		overlay_cmap: matplotlib colormap name for the overlay.

	Returns:
		[H, W, 3] uint8, oriented like `render()`'s imshow view (y increases upward).
	"""
	food = np.asarray(sim_state.env_state.food)      # [F, X, Y] bool
	walls = np.asarray(sim_state.env_state.walls)    # [X, Y] bool
	F, X, Y = food.shape

	img = np.empty((X, Y, 3), dtype=np.uint8)
	img[:] = np.asarray(background, dtype=np.uint8)

	# one colour per food type (same palette as render())
	food_colors = (np.asarray(plt.cm.Set2(np.arange(max(F, 1))))[:, :3] * 255).astype(np.uint8)  #type:ignore
	for f in range(F):
		img[food[f]] = food_colors[f]
	if walls.any():
		img[walls] = np.array([128, 128, 128], dtype=np.uint8)

	# blend the chemical field in before the agents so agents stay on top and readable
	if overlay is not None:
		o = np.asarray(overlay, dtype=np.float32)
		assert o.shape == (X, Y), f"overlay must be [X, Y] = {(X, Y)}, got {o.shape}"
		hi = float(o.max())
		if hi > 0:
			o = np.power(np.clip(o / hi, 0.0, 1.0), overlay_gamma)
			tint = (np.asarray(plt.get_cmap(overlay_cmap)(o))[..., :3] * 255).astype(np.float32)
			# alpha = intensity: empty regions keep the background, hot regions read as the tint
			a = o[..., None]
			img[:] = (img.astype(np.float32) * (1 - a) + tint * a).astype(np.uint8)

	agents = sim_state.agents_states
	alive = np.asarray(agents.alive)
	if alive.any():
		pos = np.asarray(agents.body.pos, dtype=np.float32)[alive]
		cx = np.clip(np.floor(pos[:, 0]).astype(int), 0, X - 1)
		cy = np.clip(np.floor(pos[:, 1]).astype(int), 0, Y - 1)

		if color_by == "energy":
			v = np.asarray(agents.energy, dtype=np.float32)[alive]
			v = np.clip(v / float(simulator.agent_interface.cfg.max_energy), 0.0, 1.0)
			colors = (np.asarray(plt.cm.winter(v))[:, :3] * 255).astype(np.uint8)  #type:ignore
		elif color_by == "speed":
			d = np.asarray(agents.distance_travelled, dtype=np.float32)[alive]
			age = np.clip(np.asarray(agents.age, dtype=np.float32)[alive], 1, None)
			v = d / age
			hi = float(v.max()) if v.size and v.max() > 0 else 1.0
			colors = (np.asarray(plt.cm.autumn(np.clip(v / hi, 0.0, 1.0)))[:, :3] * 255).astype(np.uint8)  #type:ignore
		elif color_by == "age":
			# normalise by max_age so the scale is fixed across frames (young=dark, old=bright)
			v = np.asarray(agents.age, dtype=np.float32)[alive]
			v = np.clip(v / max(float(simulator.agent_interface.cfg.max_age), 1.0), 0.0, 1.0)
			colors = (np.asarray(plt.cm.plasma(v))[:, :3] * 255).astype(np.uint8)  #type:ignore
		elif color_by == "nb_neurons":
			mask = getattr(agents.neural_state, "mask", None)
			if mask is None:      # non-spatial network: nothing to count, fall back to flat
				colors = np.broadcast_to(np.array([255, 255, 255], np.uint8), (cx.size, 3))
			else:
				n = np.asarray(mask).sum(-1)[alive].astype(np.float32)
				# scaled to the current population's max, since neuron counts have no fixed ceiling
				hi = float(n.max()) if n.size and n.max() > 0 else 1.0
				colors = (np.asarray(plt.cm.viridis(np.clip(n / hi, 0.0, 1.0)))[:, :3] * 255).astype(np.uint8)  #type:ignore
		else:
			colors = np.broadcast_to(np.array([255, 255, 255], np.uint8), (cx.size, 3))

		# draw a (2*agent_px+1)^2 block per agent so they survive downscaling
		for dx in range(-agent_px, agent_px + 1):
			for dy in range(-agent_px, agent_px + 1):
				img[np.clip(cx + dx, 0, X - 1), np.clip(cy + dy, 0, Y - 1)] = colors

	# match render()'s view: rows = y, cols = x, y increasing upward
	img = np.flipud(img.transpose(1, 0, 2))
	if downsample > 1:
		img = img[::downsample, ::downsample]
	return np.ascontiguousarray(img)

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
