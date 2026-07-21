"""Interactive Panel app to configure, run and watch a simulation.

Launch it from a shell or a notebook:

    from src.simulation.webapp import launch
    launch("configs/rand_baseline.yml")          # serves on http://localhost:5006

The simulation runs in this process (it needs the GPU and the local code); the browser is
just the control/view surface. Press *Start* to free-run and *Stop* to halt; the loop is a
bokeh periodic callback on the document, so there is no background thread and JAX is only
ever called from the document's event loop.

wandb logging is forced off: the app is for interactive exploration, and rebuilding a config
a dozen times should not create a dozen runs. Use `scripts/sim.py` for logged runs.
"""

import math
import time
import numpy as np
import jax, jax.numpy as jnp, jax.random as jr
import yaml
import matplotlib
matplotlib.use("Agg")          # render.py pulls in pyplot for colormaps; keep it headless
import panel as pn
from bokeh.events import Tap
from bokeh.models import ColumnDataSource, Div
from bokeh.palettes import Category10_10
from bokeh.plotting import figure

from .simulation import Simulator
from .utils import load_config_file
from .render import render_frame
from .checkpoint import save_state, load_state
from ..devo.core import Genotype
from ..settings import POSITION_DTYPE

_METRICS = ["population", "total_food", "energy (avg)", "speed (avg)", "nb_neurons (avg)",
            "age (avg)"]


def paint_food(env_state, food_type, centre, radius: float, density: float=1.0,
               erase: bool=False, *, key):
	"""Stamp a disc of food of type `food_type` onto the map, centred on `centre`.

	Backs the app's food brush, letting a food map be sculpted by hand rather than only drawn
	from `initial_density`.

	The disc uses **toroidal** distance, matching the world's wrap, so a stamp near an edge
	continues on the other side instead of being clipped. Walls are never overwritten, and food
	types stay mutually exclusive per cell (the growth step assumes at most one type per cell,
	via its `cumsum > 1` guard), so painting type `f` clears any other type underneath.

	Args:
		env_state: state to stamp onto (not mutated; a new one is returned).
		food_type: channel index to paint. May be traced.
		centre: `[2]` position in world coordinates, ordered like `Body.pos` — i.e. `centre[0]`
			indexes the food map's axis 1 and `centre[1]` its axis 2.
		radius: disc radius in cells.
		density: per-cell probability of being filled, so a stamp can be sparse rather than a
			solid blob (1.0 = solid).
		erase: clear every food type in the disc instead of painting.
		key: RNG for `density`.

	Returns:
		A new `EnvState` with the updated food map.
	"""
	F, X, Y = env_state.food.shape
	gx, gy = jnp.mgrid[:X, :Y]
	# toroidal offsets: wrap the difference so a disc straddles the seam instead of clipping
	dx = jnp.abs(gx - centre[0]); dx = jnp.minimum(dx, X - dx)
	dy = jnp.abs(gy - centre[1]); dy = jnp.minimum(dy, Y - dy)
	disc = (dx ** 2 + dy ** 2) <= radius ** 2

	sel = disc & ~env_state.walls
	if density < 1.0:
		sel = sel & jr.bernoulli(key, density, (X, Y))

	# clear first in both branches: erasing is just painting nothing, and painting must not
	# leave a second type stacked in the same cell
	food = jnp.where(sel[None], False, env_state.food)
	if not erase:
		food = food.at[food_type].set(food[food_type] | sel)
	return env_state.replace(food=food)

# steps run per periodic-callback tick at full speed. Kept small so a tick never blocks the bokeh
# event loop for long: Stop can only be handled between ticks, so this bounds how long it takes.
_STEPS_PER_TICK = 5

# floor on the callback period. Below this the event loop spends its time on callback overhead
# rather than simulating, and the UI stops responding to clicks.
_MIN_PERIOD_MS = 10

# wall-clock ceiling on the gap between UI refreshes. `record_every` counts *steps*, so on its own
# it stretches the refresh interval in proportion to how much the run is throttled — at 1 step/s
# and record_every=10 the whole UI would freeze for 10 s at a time and every click would look
# ignored. Refreshing at least this often keeps the app at ~10 fps whatever the simulation speed.
_MAX_REFRESH_GAP_MS = 100


class SimApp:
	"""Holds the simulator/state and renders the Panel view."""
	#-------------------------------------------------------------------
	def __init__(self, config_path: str="configs/rand_baseline.yml", seed: int=0,
	             image_width: int=560):
		self.config_path = config_path
		self.seed = seed
		self.image_width = image_width
		self.simulator: Simulator | None = None
		self.state = None
		self.key = None
		self._jit_step = None
		self.steps_done = 0     # plain python counter: never forces a device sync
		self._last_record = 0
		self._last_record_time = time.monotonic()
		self.running = False
		self._cb = None         # panel periodic callback driving the run loop
		self._busy = False      # guards against re-entrant / queued rebuilds
		self._runs = []         # one entry per run: {name, cds, renderers}
		self._run_counter = 0
		self._selected_idx = None   # buffer index of the inspected agent
		# the inspected agent is followed by *id*, not by buffer slot: a slot is reused by a
		# newborn as soon as its occupant dies, so tracking the index would silently switch the
		# inspector onto an unrelated agent mid-run
		self._selected_id = None
		self._jit_chem = None       # jitted chemical-field fn, built with the simulator
		self._jit_snapshot = None   # jitted per-agent gather for the inspector
		self._jit_find = None       # jitted id -> buffer-slot lookup
		# type names, replaced from the config on build; defaults keep the widgets usable before
		self._food_names: list[str] = ["food 0"]
		self._chem_names: list[str] = []
		self._obs_channels: list[str] = ["ct-1"]   # chemicals then walls, as the observation is
		self.history: dict[str, list] = {"step": [], **{m: [] for m in _METRICS}}
		self._build_widgets()
		self.load_config()
	#-------------------------------------------------------------------
	# --- model side -------------------------------------------------
	def load_config(self):
		try:
			cfg = load_config_file(self.config_path_input.value)
			self.yaml_editor.value = yaml.safe_dump(cfg, sort_keys=False)
			self._status(f"loaded <code>{self.config_path_input.value}</code> — press <b>Build &amp; Init</b>")
		except Exception as e:
			self._status(f"could not load config: <code>{e}</code>", error=True)

	def build(self):
		"""(Re)build the simulator from the YAML in the editor and initialise a fresh state."""
		cfg = yaml.safe_load(self.yaml_editor.value)
		cfg.setdefault("logging", {})["wandb_log"] = False   # never log from the app
		# drop the previous simulator/state/compiled step first: each rebuild allocates a new
		# device state and a new jit cache entry, so holding the old ones across several
		# rebuilds steadily eats GPU memory (and eventually OOMs)
		self.state = None
		self.simulator = None
		self._jit_step = None
		self._jit_chem = None
		self._jit_snapshot = None
		self._jit_find = None
		jax.clear_caches()
		simulator, _ = self._simulator_from_cfg(cfg)
		self.simulator = simulator
		self.simulator.logger = None
		# one jitted single step, compiled once and reused for every press of Step
		self._jit_step = jax.jit(lambda state, key: simulator.step(state, key=key))
		self._jit_chem = jax.jit(lambda state, key: simulator.chemical_fields(state, key=key))
		# Inspector reads, jitted. Indexing the agent buffers eagerly costs one XLA dispatch per
		# field on arrays that are large and device-sharded (W is [max_agents, N, N]), which came
		# to ~35 ms per refresh; fused into one kernel it is a fraction of that. `idx` stays traced
		# so following an agent across buffer slots never recompiles.
		def _snapshot(state, idx):
			a = state.agents_states
			net = a.neural_state
			out = {"pos": a.body.pos[idx], "id_": a.id_[idx], "generation": a.generation[idx],
			       "age": a.age[idx], "energy": a.energy[idx], "size": a.body.size[idx],
			       "n_offsprings": a.n_offsprings[idx], "distance": a.distance_travelled[idx]}
			for f in ("x", "mask", "W", "s", "m", "v"):
				leaf = getattr(net, f, None)
				if leaf is not None:
					out[f] = leaf[idx]
			return out

		def _find(state, sel_id):
			a = state.agents_states
			match = a.alive & (a.id_ == sel_id)
			return match.any(), jnp.argmax(match)

		self._jit_snapshot = jax.jit(_snapshot)
		self._jit_find = jax.jit(_find)
		self._sync_type_selectors(cfg)
		self.reset()   # sets the key from the seed widget; the config's `seed` is ignored

	def _sync_type_selectors(self, cfg: dict):
		"""Name the brush's food types and the overlay's chemical channels after the config.

		`make_world` stacks every `ft-*` / `ct-*` key in declaration order, so dict order here is
		the channel order there — the labels stay meaningful instead of being bare indices.
		"""
		food_names = [k for k in cfg if k.startswith("ft")]
		chem_names = [k for k in cfg if k.startswith("ct")]
		self._food_names = food_names or ["food 0"]
		self._chem_names = chem_names
		self.brush_type.options = self._food_names
		if self.brush_type.value not in self._food_names:
			self.brush_type.value = self._food_names[0]
		self.chem_overlay.options = ["off"] + chem_names
		if self.chem_overlay.value not in self.chem_overlay.options:
			self.chem_overlay.value = "off"
		# mini-env beacon channel: same order the observation carries — chemicals, then walls
		self._obs_channels = chem_names + ["walls"]
		self.mini_channel.options = self._obs_channels
		if self.mini_channel.value not in self._obs_channels:
			self.mini_channel.value = self._obs_channels[0]

	def _simulator_from_cfg(self, cfg: dict):
		"""Build a Simulator from an in-memory config dict (from_config_file takes a path)."""
		from .utils import make_world, make_agents_interface
		from .core import SimulationConfig
		world, _ = make_world(cfg)
		agent_interface, mutation_fn = make_agents_interface(cfg)
		simulator = Simulator(cfg=SimulationConfig(**cfg["simulation"]),
		                      world=world, agent_interface=agent_interface,
		                      mutation_fn=mutation_fn,
		                      nb_devices=cfg.get("nb_devices", None), logger=None)
		return simulator, cfg

	def reset(self):
		"""Re-initialise from the seed widget. The simulator and its compiled step are left
		alone, so switching seeds costs no recompilation, and the charts keep previous runs."""
		assert self.simulator is not None
		self.key = jr.key(int(self.seed_input.value))   # widget wins over the config's seed
		self.key, k = jr.split(self.key)
		self.state = self.simulator.initialize(key=k)
		self.steps_done = 0
		self._last_record = 0
		self._last_record_time = time.monotonic()
		# ids restart from 1, so a stale selection would latch onto an unrelated new agent
		self._selected_idx = None
		self._selected_id = None
		self._sel_source.data = {"x": [], "y": []}
		self.history = {"step": [], **{m: [] for m in _METRICS}}
		self._new_run()          # previous runs stay on the charts for comparison
		self._record()

	def _new_run(self):
		"""Start a new curve on every chart: its own source, its own colour."""
		self._run_counter += 1
		colour = Category10_10[(self._run_counter - 1) % len(Category10_10)]
		name = f"run {self._run_counter} · seed {int(self.seed_input.value)} · {colour}"
		cds = ColumnDataSource(data={"step": [], **{m: [] for m in _METRICS}})
		renderers = [f.line("step", m, source=cds, line_width=1.5, color=colour)
		             for f, m in zip(self._chart_figs, _METRICS)]
		self._runs.append({"name": name, "cds": cds, "renderers": renderers})
		self._cds = cds
		self._refresh_run_selector()

	def _delete_runs(self, names):
		"""Drop the named runs' curves from every figure."""
		doomed = [r for r in self._runs if r["name"] in set(names)]
		if not doomed:
			return
		for fig_i, fig in enumerate(self._chart_figs):
			drop = {id(r["renderers"][fig_i]) for r in doomed}
			# reassign rather than mutate in place so bokeh sees the property change
			fig.renderers = [rend for rend in fig.renderers if id(rend) not in drop]
		self._runs = [r for r in self._runs if r not in doomed]
		if not any(r["cds"] is self._cds for r in self._runs):
			self._new_run()      # the live run was deleted: start a fresh curve for it
		else:
			self._refresh_run_selector()

	# --- agent inspector ---------------------------------------------
	def _on_tap(self, event):
		"""Handle a click on the world: inspect the nearest agent, or paint/erase food."""
		try:
			if self.state is None:
				return
			if self.click_mode.value in ("paint", "erase"):
				self._brush(event.x, event.y, erase=self.click_mode.value == "erase")
				return
			if self.click_mode.value == "spawn":
				self._spawn_agent(event.x, event.y)
				return
			agents = self.state.agents_states
			alive = np.asarray(agents.alive)
			if not alive.any():
				return
			pos = np.asarray(agents.body.pos, dtype=np.float32)
			d = np.full(alive.shape, np.inf, dtype=np.float32)
			d[alive] = np.linalg.norm(pos[alive] - np.array([event.x, event.y], np.float32), axis=-1)
			self._show_agent(int(np.argmin(d)))
		except Exception as e:
			self.agent_info.text = f"<b>{type(e).__name__}</b>: <code>{e}</code>"

	def _brush(self, x: float, y: float, erase: bool=False):
		"""Stamp (or clear) a disc of food at a clicked world position.

		Runs eagerly rather than jitted: radius and density are Python-level (they branch and set
		shapes), so a jitted version would recompile on every slider move for an op that already
		costs a couple of ms.
		"""
		f_idx = self._food_names.index(self.brush_type.value) if self.brush_type.value in self._food_names else 0
		self.key, k = jr.split(self.key)
		env = paint_food(self.state.env_state, f_idx,
		                 jnp.asarray([x, y], dtype=jnp.float32),
		                 radius=float(self.brush_radius.value),
		                 density=float(self.brush_density.value),
		                 erase=erase, key=k)
		self.state = self.state.replace(env_state=env)
		self._refresh_image()
		total = float(np.asarray(self.state.env_state.food).sum())
		verb = "erased" if erase else f"painted <b>{self.brush_type.value}</b>"
		self._status(f"{verb} r={self.brush_radius.value} at ({x:.0f}, {y:.0f}) · food <b>{total:.0f}</b>")

	def _spawn_agent(self, x: float, y: float):
		"""Drop a brand-new randomly-generated agent at a clicked world position.

		Its genotype is drawn fresh (new neural params, random body size) exactly as
		`Simulator.initialize_agents` draws the founders, so it is a new lineage rather than a
		copy of anything on the map: `generation` is 1 and `parent_id_` is 0. Useful for
		reseeding after a crash, or for dropping a naive agent into an evolved population.

		The agent is written into a free (dead) slot of the fixed-size buffer, since the buffer
		cannot grow. With no free slot the population is already at `max_agents` and nothing
		happens.
		"""
		itf, sim = self.simulator.agent_interface, self.simulator
		agents = self.state.agents_states
		alive = np.asarray(agents.alive)
		free = np.flatnonzero(~alive)
		if free.size == 0:
			self._status(f"population is at max_agents ({alive.size}) — no free slot to spawn into",
			             error=True)
			return
		slot = int(free[0])

		self.key, k_prm, k_size, k_head, k_init = jr.split(self.key, 5)
		body_size = jr.uniform(k_size, (), minval=itf.cfg.min_body_size,
		                       maxval=itf.cfg.max_body_size, dtype=jnp.float16)
		# same emission signature the founders get, so a spawned agent is not chemically distinct
		sig = itf.cfg.chemical_signature
		sig = (jnp.zeros(sim.world.nb_chemicals).at[0].set(1.0) if sig is None
		       else jnp.asarray(sig, dtype=jnp.float32))
		genotype = Genotype(neural_params=itf.neural_fctry(k_prm), body_size=body_size,
		                    chemical_emission_signature=sig)
		new_id = int(np.asarray(agents.id_).max()) + 1
		new_agent = itf.init(genotype,
		                     position=jnp.asarray([x, y], dtype=POSITION_DTYPE),
		                     heading=jr.uniform(k_head, (), minval=0.0, maxval=2 * jnp.pi,
		                                        dtype=POSITION_DTYPE),
		                     id_=jnp.asarray(new_id, dtype=jnp.uint32),
		                     key=k_init)
		# write the single agent into the buffer slot, leaf by leaf
		agents = jax.tree.map(lambda buf, leaf: buf.at[slot].set(leaf), agents, new_agent)
		self.state = self.state.replace(agents_states=agents)
		self._refresh_image()
		self._status(f"spawned agent <b>{new_id}</b> at ({x:.0f}, {y:.0f}) into slot {slot} · "
		             f"population <b>{int(alive.sum()) + 1}</b>")

	def _track_selected(self):
		"""Re-resolve the inspected agent by id and redraw it. Called on every refresh.

		Looks the agent up by `id_` rather than reusing the stored buffer index, so the inspector
		follows the individual rather than the slot. Once it dies the panel freezes on its last
		state and says so, instead of silently re-pointing at whichever newborn took the slot.
		"""
		if self._selected_id is None or self.state is None:
			return
		# resolved on device: `alive` and `id_` are full max_agents buffers, so scanning them
		# host-side would copy both every refresh
		found, idx = jax.device_get(self._jit_find(self.state, jnp.uint32(self._selected_id)))
		if not bool(found):
			self._sel_source.data = {"x": [], "y": []}
			if not self.agent_info.text.startswith("<div style='color:#c0392b'>"):
				self.agent_info.text = (f"<div style='color:#c0392b'><b>agent {self._selected_id} "
				                        f"died</b> — click another to inspect</div>" + self.agent_info.text)
			self._selected_idx = None
			return
		self._show_agent(int(idx), _track=True)

	def _show_agent(self, idx: int, _track: bool=False):
		"""Draw the selected agent's grown network and list its internals."""
		# one fused kernel gathers every field for this agent, then one host transfer of the
		# small results (see `_snapshot` in build() for why the eager version was too slow)
		d = jax.device_get(self._jit_snapshot(self.state, jnp.int32(idx)))

		pos = np.asarray(d["pos"], np.float32)
		self._sel_source.data = {"x": [float(pos[0])], "y": [float(pos[1])]}
		self._selected_idx = idx
		if not _track:      # a fresh click starts following this individual
			self._selected_id = int(d["id_"])
		self.run_mini_btn.disabled = self.running

		x = np.asarray(d["x"], np.float32) if "x" in d else None
		mask = np.asarray(d["mask"]).astype(bool) if "mask" in d else None
		if x is None or mask is None:
			self.agent_info.text = "<i>this neural model is not spatially embedded</i>"
			return

		W = np.asarray(d["W"], np.float32)
		s = np.asarray(d["s"], np.float32); m = np.asarray(d["m"], np.float32)
		s1 = s.mean(-1) if s.ndim == 2 else s          # collapse gene axes to one value/neuron
		m1 = m.mean(-1) if m.ndim == 2 else m
		live = np.flatnonzero(mask)

		# connections between living neurons; W[i, j] is j -> i. Built with numpy rather than a
		# double Python loop over neurons: this now redraws on every refresh while the sim runs,
		# and an O(N^2) loop over the buffer would dominate the tick.
		e = {"x0": [], "y0": [], "x1": [], "y1": [], "color": [], "alpha": []}
		if live.size:
			sub = W[np.ix_(live, live)]
			wmax = float(np.abs(sub).max()) or 1.0
			ii, jj = np.nonzero(np.abs(sub) >= 1e-3)          # ii, jj index into `live`
			w = sub[ii, jj]
			src, dst = live[jj], live[ii]                      # j -> i
			e = {"x0": x[src, 0].tolist(), "y0": x[src, 1].tolist(),
			     "x1": x[dst, 0].tolist(), "y1": x[dst, 1].tolist(),
			     "color": np.where(w > 0, "#d62728", "#1f77b4").tolist(),
			     "alpha": (np.minimum(np.abs(w) / wmax, 1.0) * 0.6).tolist()}
		self._net_edges.data = e

		is_sensory = (s1 > 0.1) & mask
		is_motor = (m1 > 0.1) & mask
		sens, mot = is_sensory[live], is_motor[live]
		colours = np.where(sens & mot, "#9467bd",
		                   np.where(sens, "#2ca02c", np.where(mot, "#ff7f0e", "#bbbbbb")))
		sizes = 9 + 9 * np.clip(np.abs(m1[live]), 0, 1)
		self._net_nodes.data = {"x": x[live, 0].tolist(), "y": x[live, 1].tolist(),
		                        "color": colours.tolist(), "size": sizes.tolist()}

		age = max(float(d["age"]), 1.0)
		dist = float(d["distance"])
		rows = [("id", int(d["id_"])),
		        ("generation", int(d["generation"])),
		        ("age", int(age)),
		        ("energy", round(float(d["energy"]), 2)),
		        ("body size", round(float(d["size"]), 2)),
		        ("offsprings", int(d["n_offsprings"])),
		        ("distance", round(dist, 1)),
		        ("speed", round(dist / age, 3)),
		        ("neurons", int(mask.sum())),
		        ("sensory / motor", f"{int(is_sensory.sum())} / {int(is_motor.sum())}"),
		        ("synapses", len(e["x0"]))]
		# live neural activity: the one genuinely dynamic quantity in the network view, since the
		# grown structure itself is fixed for the agent's lifetime
		if "v" in d and live.size:
			v = np.asarray(d["v"], np.float32)[live]
			rows.append(("activation |v|", f"{np.abs(v).mean():.3f} (max {np.abs(v).max():.3f})"))
		cells = "".join(f"<tr><td style='padding-right:8px'>{k}</td><td><b>{v}</b></td></tr>"
		                for k, v in rows)
		self.agent_info.text = f"<table style='font-size:11px'>{cells}</table>"
		self.net_fig.title.text = f"agent {int(d['id_'])} · {int(mask.sum())} neurons"

	# --- mini-env assay -----------------------------------------------
	def _on_run_mini(self, event=None):
		"""Replay the selected agent's genotype in a controlled MiniEnv.

		Only while paused: it grows and rolls out a fresh agent, and doing that inside the run
		loop would fight the simulation for the device. The agent is re-grown from its genotype,
		so with stochastic development this is one realization of its lineage, not the exact
		individual on screen.
		"""
		try:
			if self.running:
				self.mini_info.text = "<i>pause the simulation first</i>"
				return
			if self._selected_idx is None or self.state is None:
				self.mini_info.text = "<i>click an agent first</i>"
				return
			from ..eco.mini import MiniTaxis

			idx = self._selected_idx
			agents = self.state.agents_states
			params = jax.tree.map(lambda x: x[idx], agents.genotype.neural_params)
			# reuse the agent's actual phenotype instead of re-growing it from the genotype
			grown = (jax.tree.map(lambda x: x[idx], agents.neural_state)
			         if self.mini_use_grown.value else None)
			# the interface expects sensory_genes == field channels + 4 internal signals
			n_channels = int(np.asarray(agents.neural_state.s).shape[-1]) - 4
			field = "bump" if "bump" in self.minienv_select.value else "gradient"
			steps, n_runs = int(self.mini_steps.value), int(self.mini_runs.value)
			size = int(self.mini_size.value)
			# the selector is built in observation order, so its position *is* the channel index
			chan_name = self.mini_channel.value
			channel = self._obs_channels.index(chan_name) if chan_name in self._obs_channels else 0
			if channel >= n_channels:
				self.mini_info.text = (f"<b>channel out of range</b>: <code>{chan_name}</code> is index "
				                       f"{channel} but the network only takes {n_channels} field channels")
				return
			env = MiniTaxis(self.simulator.agent_interface, grid_size=(size, size), field=field,
			                steps=steps, n_channels=n_channels, channel=channel)

			centre = np.asarray(env.grid_size, np.float32) / 2
			H, W = env.grid_size
			self.mini_fig.x_range.start, self.mini_fig.x_range.end = 0, H
			self.mini_fig.y_range.start, self.mini_fig.y_range.end = 0, W

			# Trajectories are drawn in the arena's own frame — NOT rotated to align beacons.
			# The world is clipped at its edges (a square), so any rotation would detach a
			# wall-sliding segment from the displayed boundary and make clipping look like
			# steering. Each run therefore keeps its own beacon, coloured to match its path.
			xs, ys, colours, scores, paths = [], [], [], [], []
			bx, by, bcol, bmark = [float(centre[0])], [float(centre[1])], ["#00bcd4"], ["circle"]
			first_grid = None
			for r in range(n_runs):
				self.key, k = jr.split(self.key)
				states = env.rollout(params, steps, key=k, neural_state=grown)
				pos = np.asarray(states.agent_state.body.pos, np.float32)
				src = np.asarray(states.source[0], np.float32)
				if first_grid is None:
					first_grid = np.asarray(states.state_grid[0, channel], np.float32)

				d0 = float(np.linalg.norm(centre - src))
				d1 = float(np.linalg.norm(pos[-1] - src))
				scores.append((d0 - d1) / max(d0, 1e-6))
				paths.append(float(np.linalg.norm(np.diff(pos, axis=0), axis=-1).sum()))

				colour = Category10_10[r % len(Category10_10)]
				xs.append(pos[:, 0].tolist()); ys.append(pos[:, 1].tolist())
				colours.append(colour)
				bx.append(float(src[0])); by.append(float(src[1]))
				bcol.append(colour); bmark.append("star")

			# the field differs per run (different beacon), so only show it when it is unambiguous
			blank = np.zeros((H, W), np.float32)
			self._mini_field.data = {"image": [(first_grid.T if n_runs == 1 else blank)],
			                         "dw": [H], "dh": [W]}
			self._mini_traj.data = {"xs": xs, "ys": ys, "color": colours}
			self._mini_marks.data = {"x": bx, "y": by, "color": bcol, "marker": bmark}

			sc = np.asarray(scores)
			rows = [("runs", n_runs),
			        ("beacon channel", f"{chan_name} (#{channel})"),
			        ("taxis mean", f"{sc.mean():+.3f}"),
			        ("taxis sd", f"{sc.std():.3f}"),
			        ("taxis min / max", f"{sc.min():+.2f} / {sc.max():+.2f}"),
			        ("runs with taxis &gt; 0", f"{int((sc > 0).sum())}/{n_runs}"),
			        ("mean path", f"{np.mean(paths):.1f}"),
			        ("network", "grown (as-is)" if grown is not None else "re-developed")]
			cells = "".join(f"<tr><td style='padding-right:8px'>{k}</td><td><b>{v}</b></td></tr>"
			                for k, v in rows)
			self.mini_info.text = (
				f"<table style='font-size:11px'>{cells}</table>"
				f"<div style='font-size:10px;color:#666'>★ = that run's beacon (same colour as its "
				f"path), ● = shared start. Arena is clipped at the edges, so a path can slide "
				f"along a wall. +1 = reached beacon, 0 = no progress, &lt;0 = moved away"
				f"{'' if n_runs == 1 else ' · field hidden (differs per run)'}</div>")
			self.mini_fig.title.text = (f"mini-env: {field} on {chan_name} · {n_runs} runs · "
			                            f"taxis {sc.mean():+.2f}")
		except Exception as e:
			self.mini_info.text = f"<b>{type(e).__name__}</b>: <code>{e}</code>"

	# --- checkpoints --------------------------------------------------
	def _on_save_ckpt(self, event=None):
		try:
			if self.state is None:
				self._status("nothing to save — build the simulator first", error=True)
				return
			meta = {"cfg": yaml.safe_load(self.yaml_editor.value), "step": self.steps_done,
			        "seed": int(self.seed_input.value)}
			path = save_state(self.ckpt_path.value, self.state, meta)
			self._status(f"saved checkpoint to <code>{path}</code> (step {self.steps_done})")
		except Exception as e:
			self._status(f"<b>{type(e).__name__}</b>: <code>{e}</code>", error=True)

	def _on_load_ckpt(self, event=None):
		"""Load a checkpoint into the current simulator (shapes must match its config)."""
		try:
			if self.simulator is None:
				self._status("build a simulator first, then load into it", error=True)
				return
			self._on_stop()
			state, meta = load_state(self.ckpt_path.value)
			self.state = state
			self.steps_done = int(meta.get("step", int(state.time)))
			self._last_record = self.steps_done
			self._last_record_time = time.monotonic()
			self.history = {"step": [], **{m: [] for m in _METRICS}}
			self._new_run()          # loaded state starts its own curve
			self._record()
			self._refresh()
			self._status(f"loaded <code>{self.ckpt_path.value}</code> at step {self.steps_done}")
		except Exception as e:
			self._status(f"<b>{type(e).__name__}</b>: <code>{e}</code>", error=True)

	def _refresh_run_selector(self):
		names = [r["name"] for r in self._runs]
		self.run_selector.options = names
		self.run_selector.value = [n for n in self.run_selector.value if n in names]

	def step(self, n: int, record_every: int | None=None):
		"""Advance n steps, recording metrics every `record_every` steps.

		Driven by a Python loop over the jitted single step, so nothing is recompiled when the
		step count or record interval changes. Recording periodically keeps a single press from
		producing one lonely data point; extinction is checked at those same boundaries (it
		needs a host sync anyway) and stops the loop early.
		"""
		assert self.simulator is not None and self.state is not None
		chunk = max(1, int(record_every if record_every is not None else self.record_every.value))
		n = int(n)
		for i in range(n):
			self.key, k = jr.split(self.key)
			self.state, _ = self._jit_step(self.state, k)
			self.steps_done += 1
			if (i + 1) % chunk == 0 or i == n - 1:
				self._record()
				if self.history["population"][-1] == 0:   # extinct: nothing left to simulate
					break

	def _metrics(self) -> dict:
		s = self.state
		agents = s.agents_states
		alive = np.asarray(agents.alive)
		n = int(alive.sum())
		out = {"population": n, "total_food": float(np.asarray(s.env_state.food).sum())}
		if n:
			out["energy (avg)"] = float(np.asarray(agents.energy, np.float32)[alive].mean())
			ages = np.asarray(agents.age, np.float32)[alive]
			out["age (avg)"] = float(ages.mean())
			# clip only for the speed denominator; the reported mean age is the raw one
			out["speed (avg)"] = float((np.asarray(agents.distance_travelled, np.float32)[alive]
			                            / np.clip(ages, 1, None)).mean())
			mask = getattr(agents.neural_state, "mask", None)
			out["nb_neurons (avg)"] = float(np.asarray(mask).sum(-1)[alive].mean()) if mask is not None else np.nan
		else:
			for m in ("energy (avg)", "age (avg)", "speed (avg)", "nb_neurons (avg)"):
				out[m] = np.nan
		return out

	def _record(self):
		m = self._metrics()
		self.history["step"].append(self.steps_done)
		for k in _METRICS:
			self.history[k].append(m.get(k, np.nan))
		# append to the live charts (cheap: only the new point crosses the websocket)
		self._cds.stream({"step": [self.steps_done],
		                  **{k: [m.get(k, np.nan)] for k in _METRICS}})
	#-------------------------------------------------------------------
	# --- view side --------------------------------------------------
	def _build_widgets(self):
		self.config_path_input = pn.widgets.TextInput(name="config file", value=self.config_path)
		self.load_btn = pn.widgets.Button(name="Load config", button_type="default")
		self.yaml_editor = pn.widgets.TextAreaInput(name="config (editable)", height=300,
		                                            sizing_mode="stretch_width")
		self.build_btn = pn.widgets.Button(name="Build & Init", button_type="primary")
		self.reset_btn = pn.widgets.Button(name="Reset", button_type="warning")
		# seed is applied on Reset and overrides the one in the config: changing it re-runs with
		# a different RNG stream without touching the simulator, so nothing is recompiled
		self.seed_input = pn.widgets.IntInput(name="seed (applied on Reset)", value=0, step=1)
		self.run_selector = pn.widgets.MultiSelect(name="runs on charts", options=[], size=5,
		                                           sizing_mode="stretch_width")
		self.delete_runs_btn = pn.widgets.Button(name="Delete selected", button_type="warning")
		self.clear_runs_btn = pn.widgets.Button(name="Clear all", button_type="danger")
		self.record_every = pn.widgets.IntInput(name="refresh every (steps)", value=10, step=10,
		                                        start=1, end=10000)
		self.start_btn = pn.widgets.Button(name="▶ Start", button_type="success")
		self.stop_btn = pn.widgets.Button(name="■ Stop", button_type="danger", disabled=True)
		# --- speed. Full speed simulates as fast as the device manages; throttling paces the run
		# so slow dynamics stay watchable, and leaves the event loop idle between ticks.
		self.full_speed = pn.widgets.Checkbox(name="full speed (no wait)", value=True)
		self.target_sps = pn.widgets.IntSlider(name="target steps/s", value=20, start=1, end=500,
		                                       step=1)
		self.render_image = pn.widgets.Checkbox(name="render world while running", value=True)
		self.color_by = pn.widgets.Select(name="colour agents by",
		                                  options=["energy", "speed", "age", "nb_neurons", "flat"],
		                                  value="energy")
		self.agent_px = pn.widgets.IntSlider(name="agent size (px)", value=1, start=0, end=4)
		# --- chemical overlay: off by default, since computing the field costs a diffusion
		# convolution per refresh and the plain view is the one you want most of the time
		self.chem_overlay = pn.widgets.Select(name="chemical overlay", options=["off"], value="off")
		self.chem_gamma = pn.widgets.FloatSlider(name="overlay contrast (gamma)", value=0.5,
		                                         start=0.1, end=1.5, step=0.05)
		# --- food brush: click the world to stamp a disc of food (or erase one)
		self.click_mode = pn.widgets.RadioButtonGroup(
			name="click action", options=["inspect", "paint", "erase", "spawn"], value="inspect")
		self.brush_type = pn.widgets.Select(name="brush food type", options=["food 0"],
		                                    value="food 0")
		self.brush_radius = pn.widgets.IntSlider(name="brush radius (cells)", value=8, start=1,
		                                         end=64)
		self.brush_density = pn.widgets.FloatSlider(name="brush density", value=1.0, start=0.02,
		                                            end=1.0, step=0.02)
		# counter / status / world are raw bokeh models rather than panel panes: bokeh syncs
		# model properties incrementally over the websocket, which stays responsive over an ssh
		# tunnel, whereas a PNG pane re-sends the whole image base64-encoded on every update.
		self.counter = Div(text="<h3>step 0</h3>", sizing_mode="stretch_width")
		self.status = Div(text="", sizing_mode="stretch_width")
		self._img_source = ColumnDataSource(data={"image": [np.zeros((1, 1), dtype=np.uint32)],
		                                          "x": [0], "y": [0], "dw": [1], "dh": [1]})
		# grows with the available space but keeps the world's proportions: ranges and
		# aspect_ratio are set from the grid dimensions in _refresh_image, so a non-square
		# world is not stretched. Ranges are in world cells, so tap coords are world coords.
		self.image = figure(sizing_mode="scale_both", aspect_ratio=1,
		                    x_range=(0, 1), y_range=(0, 1), toolbar_location=None,
		                    match_aspect=True)
		self.image.axis.visible = False
		self.image.grid.visible = False
		self.image.image_rgba(image="image", x="x", y="y", dw="dw", dh="dh",
		                      source=self._img_source)
		# click the world to inspect the nearest agent (ranges are in world cells, so the tap
		# event's data coordinates are world coordinates directly)
		self.image.on_event(Tap, self._on_tap)
		# ring following the tracked agent. Amber and thick so it stays findable against both the
		# food palette and the chemical overlay while the agent moves.
		self._sel_source = ColumnDataSource(data={"x": [], "y": []})
		self.image.scatter("x", "y", source=self._sel_source, size=16, line_color="#ffc107",
		                   fill_alpha=0.0, line_width=3)

		# --- agent inspector: grown network + internals ---
		self._net_edges = ColumnDataSource(data={"x0": [], "y0": [], "x1": [], "y1": [],
		                                         "color": [], "alpha": []})
		self._net_nodes = ColumnDataSource(data={"x": [], "y": [], "color": [], "size": []})
		self.net_fig = figure(sizing_mode="stretch_width", height=300, aspect_ratio=1,
		                      x_range=(-1.1, 1.1), y_range=(-1.1, 1.1),
		                      title="agent network (click an agent)", toolbar_location=None)
		self.net_fig.axis.visible = False
		self.net_fig.grid.visible = False
		self.net_fig.title.text_font_size = "9pt"
		self.net_fig.segment("x0", "y0", "x1", "y1", source=self._net_edges,
		                     line_color="color", line_alpha="alpha", line_width=1)
		self.net_fig.scatter("x", "y", source=self._net_nodes, size="size", color="color",
		                     line_color="black", line_width=0.5)
		self.agent_info = Div(text="<i>click an agent in the world to inspect it</i>",
		                      sizing_mode="stretch_width")

		# --- mini-env assay: replay the selected agent in a controlled environment ---
		self.minienv_select = pn.widgets.Select(
			name="mini-env", value="taxis · gradient",
			options=["taxis · gradient", "taxis · bump"])
		# which observation channel the beacon is emitted into. Options are filled on build in the
		# simulation's own channel order (chemicals, then walls), so the label matches the index.
		self.mini_channel = pn.widgets.Select(name="beacon emits on", options=["ct-1"],
		                                      value="ct-1")
		self.mini_steps = pn.widgets.IntInput(name="mini-env steps", value=200, step=50,
		                                      start=10, end=5000)
		self.mini_runs = pn.widgets.IntInput(name="mini-env runs (seeds)", value=5, step=1,
		                                     start=1, end=50)
		# grid size of the assay world; the beacon sits at 80% of the way to the nearest edge,
		# so a bigger grid means a proportionally longer trip to it
		self.mini_size = pn.widgets.IntInput(name="mini-env size (cells)", value=64, step=16,
		                                     start=16, end=512)
		self.mini_use_grown = pn.widgets.Checkbox(
			name="use grown network (skip development)", value=True)
		self.run_mini_btn = pn.widgets.Button(name="Replay agent in mini-env",
		                                      button_type="primary", disabled=True)
		self._mini_field = ColumnDataSource(data={"image": [np.zeros((2, 2), np.float32)],
		                                          "dw": [1], "dh": [1]})
		self._mini_traj = ColumnDataSource(data={"xs": [], "ys": [], "color": []})
		self._mini_marks = ColumnDataSource(data={"x": [], "y": [], "color": [], "marker": []})
		self.mini_fig = figure(sizing_mode="stretch_width", height=300, aspect_ratio=1,
		                       title="mini-env replay", toolbar_location=None,
		                       match_aspect=True)
		self.mini_fig.axis.visible = False
		self.mini_fig.grid.visible = False
		self.mini_fig.title.text_font_size = "9pt"
		self.mini_fig.image(image="image", x=0, y=0, dw="dw", dh="dh",
		                    palette="Viridis256", source=self._mini_field)
		self.mini_fig.multi_line("xs", "ys", source=self._mini_traj, line_color="color",
		                         line_width=2, line_alpha=0.9)
		self.mini_fig.scatter("x", "y", source=self._mini_marks, size=13, color="color",
		                      marker="marker", line_color="white")
		self.mini_info = Div(text="", sizing_mode="stretch_width")

		# --- checkpoints ---
		self.ckpt_path = pn.widgets.TextInput(name="checkpoint file", value="data/checkpoint.ckpt")
		self.save_ckpt_btn = pn.widgets.Button(name="Save ckpt", button_type="default")
		self.load_ckpt_btn = pn.widgets.Button(name="Load ckpt", button_type="default")
		# streaming charts. Each run gets its own ColumnDataSource and its own line on every
		# figure, so Reset/Build start a new curve instead of wiping the old ones and runs can
		# be compared. Bokeh streams only the new point per record.
		self._cds = None            # current run's source, created by _new_run()
		figs = []
		for m in _METRICS:
			f = figure(sizing_mode="stretch_both", min_height=110, title=m,
			           toolbar_location=None, output_backend="webgl")
			f.title.text_font_size = "9pt"
			if figs:
				f.x_range = figs[0].x_range      # pan/zoom the panels together
			figs.append(f)
		self._chart_figs = figs
		figs[-1].xaxis.axis_label = "step"
		# wrap each figure in a Bokeh pane that also stretches: panel wraps raw bokeh models
		# automatically, but the implicit wrapper is fixed-size and would pin the layout
		self.charts = pn.Column(*[pn.pane.Bokeh(f, sizing_mode="stretch_both") for f in figs],
		                        sizing_mode="stretch_both")

		self.load_btn.on_click(lambda e: self.load_config())
		self.build_btn.on_click(self._on_build)
		self.reset_btn.on_click(self._on_reset)
		self.run_mini_btn.on_click(self._on_run_mini)
		self.save_ckpt_btn.on_click(self._on_save_ckpt)
		self.load_ckpt_btn.on_click(self._on_load_ckpt)
		self.delete_runs_btn.on_click(lambda e: self._delete_runs(list(self.run_selector.value)))
		self.clear_runs_btn.on_click(lambda e: self._delete_runs([r["name"] for r in self._runs]))
		self.start_btn.on_click(self._on_start)
		self.stop_btn.on_click(self._on_stop)
		self.color_by.param.watch(lambda e: self._refresh_image(), "value")
		self.agent_px.param.watch(lambda e: self._refresh_image(), "value")
		self.chem_overlay.param.watch(lambda e: self._refresh_image(), "value")
		self.chem_gamma.param.watch(lambda e: self._refresh_image(), "value")
		self.full_speed.param.watch(self._restart_loop, "value")
		self.target_sps.param.watch(self._restart_loop, "value")

	def _guard(self, fn):
		"""Run a callback, surfacing errors in the status bar instead of the console."""
		try:
			fn()
		except Exception as e:
			self._status(f"<b>{type(e).__name__}</b>: <code>{e}</code>", error=True)

	def _on_build(self, event=None):
		"""Disable the controls, then do the (slow, blocking) build on the next tick.

		Building compiles a fresh jitted step, which takes tens of seconds and blocks bokeh's
		event loop. Doing it inline would leave the buttons live, so every extra click queues
		another full rebuild. Deferring by one tick lets the disabled state reach the browser
		first, making further clicks impossible.
		"""
		if self._busy:
			return
		self._on_stop()
		self._busy = True
		self._set_busy_ui(True)
		self._status("building… (compiles the step function, this can take ~30 s)")
		# schedule on the bokeh document: doc callbacks run holding the document lock, whereas
		# panel's PeriodicCallback either hands sync callbacks to an unlocked worker thread or
		# `await`s async ones (which releases the lock) -> "we should have the lock" errors
		doc = pn.state.curdoc
		if doc is not None:
			doc.add_next_tick_callback(self._do_build)
		else:
			self._do_build()          # headless (tests / notebook without a server)

	def _do_build(self):
		# Sync, and scheduled via doc.add_next_tick_callback: bokeh runs document callbacks
		# while holding the document lock, so the widget writes below are safe.
		try:
			self.build()
			self._refresh()
		except Exception as e:
			self._status(f"<b>{type(e).__name__}</b>: <code>{e}</code>", error=True)
		finally:
			self._busy = False
			self._set_busy_ui(False)

	def _on_reset(self, event=None):
		if self._busy:
			return
		if self.simulator is None:
			self._status("build the simulator first", error=True)
			return
		self._on_stop()
		self.reset()
		self._refresh()

	def _set_busy_ui(self, busy: bool):
		self.build_btn.disabled = busy
		self.reset_btn.disabled = busy
		self.start_btn.disabled = busy
		self.load_btn.disabled = busy

	def _tick_params(self) -> tuple[int, int]:
		"""(steps per tick, callback period in ms) for the current speed setting.

		The period cannot go below `_MIN_PERIOD_MS`, so rates above that floor are reached by
		simulating several steps per tick rather than ticking faster. Stop is only serviced
		between ticks, so this also keeps the button responsive at any speed.
		"""
		if self.full_speed.value:
			return _STEPS_PER_TICK, _MIN_PERIOD_MS
		target = max(1, int(self.target_sps.value))
		steps = max(1, math.ceil(target * _MIN_PERIOD_MS / 1000))
		return steps, max(_MIN_PERIOD_MS, round(1000 * steps / target))

	def _restart_loop(self, *_):
		"""Re-register the run loop so a speed change takes effect immediately, not on next Start."""
		if not self.running:
			return
		doc = pn.state.curdoc
		if doc is None or self._cb is None:
			return
		try:
			doc.remove_periodic_callback(self._cb)
		except (ValueError, RuntimeError):
			pass
		_, period = self._tick_params()
		self._cb = doc.add_periodic_callback(self._tick, period)

	def _on_start(self, event=None):
		"""Free-run the simulation via a Panel periodic callback until Stop (or extinction).

		Each tick is a short, synchronous callback that Bokeh runs while holding the document
		lock. An async loop that `await`s instead would release that lock and then write to the
		document on resume, which raises "we should have the lock when the document changes".
		"""
		if self.state is None:
			self._status("build the simulator first", error=True)
			return
		if self.running:
			return
		self.running = True
		self._set_running_ui(True)
		doc = pn.state.curdoc
		if doc is not None:
			_, period = self._tick_params()
			self._cb = doc.add_periodic_callback(self._tick, period)   # runs with the doc lock held

	def _on_stop(self, event=None):
		"""Stop the run loop and draw a final frame."""
		if self._cb is not None:
			doc = pn.state.curdoc
			try:
				if doc is not None:
					doc.remove_periodic_callback(self._cb)
			except (ValueError, RuntimeError):
				pass
			self._cb = None
		was_running = self.running
		self.running = False
		self._set_running_ui(False)
		if was_running:
			# turning rendering off during the run should still leave a current world on screen
			self._refresh_image()

	def _tick(self):
		"""A short batch of steps plus a UI update.

		Sync, and registered with doc.add_periodic_callback, so bokeh runs it holding the
		document lock (panel's own PeriodicCallback would either hand a sync callback to an
		unlocked worker thread or `await` an async one, releasing the lock).

		Deliberately only `_STEPS_PER_TICK` steps: the tick blocks bokeh's event loop, so Stop
		can only be serviced between ticks, and running `record_every` steps here would make the
		button lag by that whole batch. Metrics are recorded every `record_every` steps *across*
		ticks instead, so the two concerns stay independent.
		"""
		if not self.running:
			return
		try:
			steps_per_tick, _ = self._tick_params()
			for _ in range(steps_per_tick):
				self.key, k = jr.split(self.key)
				self.state, _ = self._jit_step(self.state, k)
				self.steps_done += 1          # plain python counter: no device sync
			# jax dispatch is async, so without this the python loop races ahead queueing work
			# and a later sync blocks for a long time — which is what made Stop feel frozen
			jax.block_until_ready(self.state.time)
			self.counter.text = f"<h3>step {self.steps_done}</h3>"

			# refresh on whichever comes first: `record_every` steps, or _MAX_REFRESH_GAP_MS of
			# wall time. The time trigger requires at least one new step, so a slow run animates
			# every step instead of stamping duplicate points onto the charts.
			now = time.monotonic()
			due_steps = self.steps_done - self._last_record >= max(1, int(self.record_every.value))
			due_time = (self.steps_done > self._last_record
			            and (now - self._last_record_time) * 1000 >= _MAX_REFRESH_GAP_MS)
			if due_steps or due_time:
				self._last_record = self.steps_done
				self._last_record_time = now
				self._record()                # also streams the new point to the charts
				if self.render_image.value:   # rendering is the costly part of a refresh
					self._refresh_image()
				# tracked independently of render_image: the marker and the inspector are their
				# own glyphs, so following an agent still works with the world view turned off
				self._track_selected()
				self._status_line()
				if self.history["population"][-1] == 0:
					self._on_stop()
					self._status("population <b>extinct</b> — stopped", error=True)
		except Exception as e:
			self._on_stop()
			self._status(f"<b>{type(e).__name__}</b>: <code>{e}</code>", error=True)

	def _set_running_ui(self, running: bool):
		self.start_btn.disabled = running
		self.stop_btn.disabled = not running
		# the mini-env replay grows and rolls out an agent, so only offer it while paused
		self.run_mini_btn.disabled = running or self._selected_idx is None
		self.build_btn.disabled = running
		self.reset_btn.disabled = running

	def _status(self, msg: str, error: bool=False):
		colour = "#c0392b" if error else "#2c3e50"
		self.status.text = f"<span style='color:{colour}'>{msg}</span>"

	def _chem_overlay_field(self) -> np.ndarray | None:
		"""The selected chemical channel as an [X, Y] host array, or None when the overlay is off.

		Off by default and skipped entirely when off: this runs the diffusion convolution and
		pulls a full grid back to the host, which would otherwise be paid on every refresh.
		"""
		name = self.chem_overlay.value
		if name == "off" or self._jit_chem is None or name not in self._chem_names:
			return None
		self.key, k = jr.split(self.key)
		fields = self._jit_chem(self.state, k)          # [C, X, Y]
		return np.asarray(fields[self._chem_names.index(name)], dtype=np.float32)

	def _refresh_image(self):
		if self.state is None or self.simulator is None:
			return
		img = render_frame(self.simulator, self.state,
		                   color_by=self.color_by.value, agent_px=int(self.agent_px.value),
		                   overlay=self._chem_overlay_field(),
		                   overlay_gamma=float(self.chem_gamma.value))
		# pack RGB -> uint32 RGBA for image_rgba (sent as binary, not base64). render_frame
		# returns row 0 = top (imshow convention); image_rgba draws row 0 at the bottom, so flip
		h, w, _ = img.shape          # h = world rows (y), w = world cols (x)
		rgba = np.dstack([img, np.full((h, w, 1), 255, np.uint8)])
		self._img_source.data = {"image": [np.flipud(rgba.view(np.uint32).reshape(h, w))],
		                         "x": [0], "y": [0], "dw": [w], "dh": [h]}
		# keep the plot proportional to the grid: a 256x512 world must not be drawn square
		if self.image.x_range.end != w or self.image.y_range.end != h:
			self.image.x_range.start, self.image.x_range.end = 0, w
			self.image.y_range.start, self.image.y_range.end = 0, h
			self.image.aspect_ratio = w / h

	def _status_line(self):
		"""Status from the last recorded metrics, so it costs no extra device sync."""
		if not self.history["step"]:
			return
		pop, food = self.history["population"][-1], self.history["total_food"][-1]
		extinct = " — <b>extinct</b>" if pop == 0 else ""
		self._status(f"population <b>{pop}</b> · food <b>{food:.0f}</b>{extinct}")

	def _refresh(self):
		# charts need no explicit redraw: they stream from _record()
		self._refresh_image()
		self._track_selected()
		self._status_line()
		self.counter.text = f"<h3>step {self.steps_done}</h3>"

	def view(self):
		"""Responsive three-column layout: fixed-ish controls, then world and charts sharing
		the remaining width. Everything stretches, so it fills a large screen instead of
		collapsing into the top-left corner."""
		controls = pn.Column(
			pn.pane.Markdown("### EcoEvoDevox"),
			self.config_path_input, self.load_btn,
			self.yaml_editor,
			pn.Row(self.build_btn, self.reset_btn, sizing_mode="stretch_width"),
			self.seed_input,
			pn.Row(self.start_btn, self.stop_btn, sizing_mode="stretch_width"),
			self.full_speed, self.target_sps,
			self.record_every,
			self.render_image, self.color_by, self.agent_px,
			pn.pane.Markdown("**chemicals**", margin=(6, 0, -6, 0)),
			self.chem_overlay, self.chem_gamma,
			pn.pane.Markdown("**food brush** — click the world", margin=(6, 0, -6, 0)),
			self.click_mode, self.brush_type, self.brush_radius, self.brush_density,
			self.run_selector,
			pn.Row(self.delete_runs_btn, self.clear_runs_btn, sizing_mode="stretch_width"),
			self.ckpt_path,
			pn.Row(self.save_ckpt_btn, self.load_ckpt_btn, sizing_mode="stretch_width"),
			width=360, sizing_mode="stretch_height", scroll=True,
		)
		world = pn.Column(
			pn.pane.Bokeh(self.counter, sizing_mode="stretch_width"),
			pn.pane.Bokeh(self.status, sizing_mode="stretch_width"),
			pn.pane.Bokeh(self.image, sizing_mode="stretch_both"),
			sizing_mode="stretch_both")
		inspector = pn.Column(
			pn.pane.Bokeh(self.net_fig, sizing_mode="stretch_width"),
			pn.pane.Bokeh(self.agent_info, sizing_mode="stretch_width"),
			self.minienv_select, self.mini_channel, self.mini_size, self.mini_steps, self.mini_runs,
			self.mini_use_grown, self.run_mini_btn,
			pn.pane.Bokeh(self.mini_fig, sizing_mode="stretch_width"),
			pn.pane.Bokeh(self.mini_info, sizing_mode="stretch_width"),
			width=340, sizing_mode="stretch_height", scroll=True)
		return pn.Row(controls, world, inspector, self.charts,
		              sizing_mode="stretch_both")


def launch(config_path: str="configs/rand_baseline.yml", port: int=5006,
           show: bool=True, **serve_kwargs):
	"""Serve the app at http://localhost:<port>."""
	pn.extension()
	app = SimApp(config_path=config_path)
	return pn.serve(app.view(), port=port, show=show, **serve_kwargs)


if __name__ == "__main__":
	launch()
