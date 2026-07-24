"""Minimal, single-agent environments for probing an `AgentInterface` in isolation.

`MiniEnv` is a lightweight mirror of the full `Simulator` loop for **one** agent on a
small static grid: grow the agent, sample a sensory field at its body points, step the
sensorimotor loop, move. It carries none of the population/food/energy machinery, so it is
cheap to `vmap`/`scan` over and is meant for QD scoring, unit tests and quick behavioural
probes. Subclass it and implement `reset` (build the grid + agent) and `evaluate` (a
scalar objective); everything else — observation, step, rollout — is shared.

`MiniTaxis` is the first concrete env: chemotaxis toward a crafted chemical beacon.

This mirrors the observation path of `Simulator.step_agents`: the sensory field is sampled
at the agent's body points and handed to `AgentInterface.step`, which prepends the 4
internal signals itself. So a spatially-embedded sensory model needs
`sensory_genes == n_field_channels + 4`.
"""

import jax
import numpy as np
from jax import numpy as jnp, random as jr
from flax.struct import PyTreeNode
from jaxtyping import Float

from ..devo.core import AgentState, Body, Genotype, NeuralParams
from ..devo.interface import AgentInterface
from ..settings import POSITION_DTYPE
from .gridworld import get_cell_index


class MiniEnvState(PyTreeNode):
	state_grid: jax.Array   # [C, H, W] static sensory field(s) the agent perceives
	agent_state: AgentState


class MiniEnv:
	"""Single-agent environment on a static [C, H, W] grid. Subclass `reset`/`evaluate`."""
	#-------------------------------------------------------------------
	def __init__(self, grid_size: tuple[int, int], agent_interface: AgentInterface,
	             body_size: float = 2.0):
		self.grid_size = grid_size
		self.agent_interface = agent_interface
		self.body_size = body_size
	#-------------------------------------------------------------------
	def make_genotype(self, params: NeuralParams) -> Genotype:
		"""Wrap neural parameters into a Genotype. The chemical emission signature is unused
		here (agents do not emit in a MiniEnv), so it is a placeholder."""
		return Genotype(neural_params=params,
		                body_size=jnp.asarray(self.body_size, dtype=jnp.float16),
		                chemical_emission_signature=jnp.zeros(1))
	#-------------------------------------------------------------------
	def init_agent_state(self, genotype: Genotype, key: jax.Array,
	                     neural_state=None) -> AgentState:
		"""Place an agent at the grid centre with a random heading.

		By default the network is grown from `genotype`. Pass `neural_state` to reuse an
		already-grown network instead — e.g. the exact phenotype an agent has in a running
		simulation — which skips development entirely (useful when development is stochastic
		and you want *this* individual rather than a fresh draw from its lineage). Only the
		sensory/motor layouts are rebuilt, since those are caches derived from the network.
		"""
		k_head, k_init = jr.split(key)
		start_pos = jnp.asarray(self.grid_size, dtype=POSITION_DTYPE) / 2
		start_heading = jr.uniform(k_head, (), minval=0.0, maxval=2 * jnp.pi, dtype=POSITION_DTYPE)
		if neural_state is None:
			return self.agent_interface.init(
				genotype, position=start_pos, heading=start_heading,
				id_=jnp.ones((), dtype=jnp.uint32), key=k_init,
			)

		itf = self.agent_interface
		k_sens, k_mot = jr.split(k_init)
		body_size = jnp.clip(genotype.body_size, itf.cfg.min_body_size,
		                     itf.cfg.max_body_size).astype(itf.cfg.min_body_size.dtype)
		return AgentState(
			genotype=genotype.replace(body_size=body_size),
			body=Body(pos=start_pos, heading=start_heading, size=body_size),
			motor_state=itf._motor_interface.init(neural_state, key=k_mot),
			sensory_state=itf._sensory_interface.init(neural_state, k_sens),
			neural_state=neural_state,
			alive=jnp.ones((), dtype=jnp.bool),
			age=jnp.ones((), dtype=jnp.uint16),
			energy=itf.cfg.init_energy,
			time_above_threshold=jnp.zeros((), dtype=jnp.uint16),
			time_below_threshold=jnp.zeros((), dtype=jnp.uint16),
			n_offsprings=jnp.zeros((), dtype=jnp.uint16),
			distance_travelled=jnp.zeros((), dtype=jnp.float32),
			total_abs_turn=jnp.zeros((), dtype=jnp.float32),
			generation=jnp.ones((), dtype=jnp.uint32),
			id_=jnp.ones((), dtype=jnp.uint32),
			parent_id_=jnp.zeros((), dtype=jnp.uint32),
		)
	#-------------------------------------------------------------------
	def reset(self, params: NeuralParams, key: jax.Array, neural_state=None) -> MiniEnvState:
		"""Build the initial state (sensory grid + agent). Subclass responsibility.

		Subclasses should forward `neural_state` to `init_agent_state` so callers can reuse an
		already-grown network instead of re-running development.
		"""
		raise NotImplementedError
	#-------------------------------------------------------------------
	def get_observation(self, state: MiniEnvState) -> jax.Array:
		"""Sample the sensory field at the agent's body points -> [C, R, R].

		Returns the raw env array; `AgentInterface.step` prepends the internal signals and
		builds the `Observation`, exactly as the full simulator does.
		"""
		body = state.agent_state.body
		cells = get_cell_index(self.agent_interface.get_body_points(body))  # [2, R, R]
		H, W = self.grid_size
		i = jnp.clip(cells[0], 0, H - 1)
		j = jnp.clip(cells[1], 0, W - 1)
		return state.state_grid[:, i, j]  # [C, R, R]
	#-------------------------------------------------------------------
	def step(self, state: MiniEnvState, key: jax.Array) -> MiniEnvState:
		obs = self.get_observation(state)
		# step now moves the body itself (actuate); the mini world is bounded (not toroidal), so we
		# clip the moved body into the grid here instead of the simulator's toroidal wrap
		agent_state, _ = self.agent_interface.step(obs, state.agent_state, key)
		bound = jnp.asarray(self.grid_size, dtype=agent_state.body.pos.dtype) - jnp.asarray(1e-3, agent_state.body.pos.dtype)
		new_body = agent_state.body.replace(pos=jnp.clip(agent_state.body.pos, 0.0, bound))
		return state.replace(agent_state=agent_state.replace(body=new_body))
	#-------------------------------------------------------------------
	def rollout(self, params: NeuralParams, steps: int, key: jax.Array,
	            neural_state=None) -> MiniEnvState:
		"""Run `steps` steps; returns the stacked post-step states (so [-1] is the final one).

		`neural_state` is forwarded to `reset` to reuse an already-grown network.
		"""
		key_init, key_roll = jr.split(key)
		state = self.reset(params, key_init, neural_state=neural_state)

		def _step(state, k):
			new_state = self.step(state, k)
			return new_state, new_state

		_, states = jax.lax.scan(_step, state, jr.split(key_roll, steps))
		return states
	#-------------------------------------------------------------------
	def evaluate(self, params: NeuralParams, key: jax.Array) -> tuple[Float, dict]:
		"""Roll out and return (scalar fitness, info dict). Subclass responsibility."""
		raise NotImplementedError


class MiniTaxisState(MiniEnvState):
	source: jax.Array   # [2] beacon (field maximum) position, in grid coordinates


class MiniTaxis(MiniEnv):
	"""Chemotaxis toward a crafted chemical beacon.

	A single chemical field peaks at a beacon placed at a random bearing, a fixed `radius`
	from the grid centre where the agent starts. The agent senses the field through its
	body (via the spatially-embedded interface), and fitness rewards ending the rollout
	closer to the beacon than it started:  ``(d_start - d_end) / d_start``  in [-inf, 1],
	1 meaning it reached the beacon. Because the beacon bearing is drawn from `key`, scoring
	a genotype over several keys (e.g. `nb_evals`) rewards general taxis, not one direction.

	`field`:
		"gradient" — concentration falls off linearly with distance to the beacon, so there
			is a usable directional signal *everywhere* (recommended: evolution is not stuck
			at zero waiting to stumble onto a local bump).
		"bump"     — a Gaussian centred on the beacon (width `sigma`); only sensable nearby.

	Note the field channel count is 1, so a spatially-embedded sensory model needs
	`sensory_genes == 1 + 4` (field + internals).
	"""
	#-------------------------------------------------------------------
	def __init__(self, agent_interface: AgentInterface, grid_size: tuple[int, int] = (32, 32),
	             body_size: float = 2.0, field: str = "gradient", radius: float | None = None,
	             sigma: float = 5.0, steps: int = 32, n_channels: int = 1, channel: int = 0):
		"""`n_channels` sets how many field channels the observation carries; all but `channel`
		are zeros. This lets an agent taken from the full simulation be dropped in here unchanged:
		a spatially-embedded interface needs `sensory_genes == n_channels + 4` (the 4 internal
		signals), so pass the sim's channel count (n_chemicals + walls) rather than the default 1.

		`channel` picks **which** input field carries the beacon. The full simulation orders
		observation channels as the chemicals in `ct-*` declaration order, then walls last (see
		`GridWorld.get_agents_observations`), so `channel=0` is `ct-1` and `channel=n_channels-1`
		is the wall channel. Choosing it matters because an agent's sensor genes are tuned to a
		specific channel: probing taxis on the wrong one tests a sense the agent never evolved."""
		super().__init__(grid_size, agent_interface, body_size)
		assert field in ("gradient", "bump"), f"field must be 'gradient' or 'bump', got {field!r}"
		self.field = field
		self.channel = int(channel)
		# default beacon radius: 80% of the way from centre to the nearest edge
		self.radius = radius if radius is not None else 0.8 * min(grid_size) / 2
		self.sigma = sigma
		self.steps = steps
		self.n_channels = max(1, int(n_channels))
		assert 0 <= self.channel < self.n_channels, (
			f"channel {self.channel} out of range for {self.n_channels} observation channel(s)")
	#-------------------------------------------------------------------
	def chem_field(self, source: jax.Array) -> jax.Array:
		"""Static concentration field peaking at `source` -> [n_channels, H, W].

		The beacon occupies `self.channel`; every other channel is zero."""
		H, W = self.grid_size
		grid = jnp.mgrid[:H, :W].astype(jnp.float32)          # [2, H, W]
		dist = jnp.linalg.norm(grid - source[:, None, None], axis=0)  # [H, W]
		if self.field == "gradient":
			max_dist = jnp.linalg.norm(jnp.asarray(self.grid_size, jnp.float32))
			conc = 1.0 - dist / max_dist
		else:  # bump
			conc = jnp.exp(-dist ** 2 / (2 * self.sigma ** 2))
		return jnp.zeros((self.n_channels, H, W), conc.dtype).at[self.channel].set(conc)
	#-------------------------------------------------------------------
	def reset(self, params: NeuralParams, key: jax.Array, neural_state=None) -> MiniTaxisState:
		k_agent, k_beacon = jr.split(key)
		agent_state = self.init_agent_state(self.make_genotype(params), k_agent,
		                                    neural_state=neural_state)
		# beacon at a random bearing, `radius` from the centre (so d_start == radius always)
		angle = jr.uniform(k_beacon, (), minval=0.0, maxval=2 * jnp.pi)
		center = jnp.asarray(self.grid_size, jnp.float32) / 2
		source = center + self.radius * jnp.array([jnp.cos(angle), jnp.sin(angle)])
		return MiniTaxisState(self.chem_field(source), agent_state, source)
	#-------------------------------------------------------------------
	def evaluate(self, params: NeuralParams, key: jax.Array) -> tuple[Float, dict]:
		states: MiniTaxisState = self.rollout(params, self.steps, key)
		source = states.source[0].astype(jnp.float32)                 # constant over the rollout
		start = jnp.asarray(self.grid_size, jnp.float32) / 2          # agents always start centred
		final = states.agent_state.body.pos[-1].astype(jnp.float32)

		d_start = jnp.linalg.norm(start - source)
		d_end = jnp.linalg.norm(final - source)
		fitness = (d_start - d_end) / jnp.clip(d_start, 1e-6)

		# diagnostic: mean concentration the agent sat in along the way
		field = states.state_grid[0, self.channel]                   # [H, W]
		pos = states.agent_state.body.pos.astype(jnp.float32)        # [steps, 2]
		cells = jnp.clip(jnp.floor(pos).astype(jnp.int32), 0,
		                 jnp.asarray(self.grid_size, jnp.int32) - 1)
		mean_conc = field[cells[:, 0], cells[:, 1]].mean()

		return fitness, {"d_start": d_start, "d_end": d_end, "mean_conc": mean_conc}


class MiniMultiTaxisState(MiniTaxisState):
	n_reached: jax.Array   # beacons reached so far
	dist_sum: jax.Array    # running sum of distance to the beacon that was live at each step


class MiniMultiTaxis(MiniTaxis):
	"""Sequential chemotaxis: reach a beacon, another appears somewhere else.

	Exactly one beacon exists at a time. When the agent gets within `reach_threshold` of it, the
	beacon is consumed, the counter increments and a fresh one is drawn elsewhere. Fitness is the
	number of beacons reached over the rollout.

	This asks for something `MiniTaxis` cannot: a single-beacon run rewards drifting in one lucky
	direction, whereas here an agent must re-orient to a *new* bearing each time, so a fixed
	turning bias scores once and then stops paying. It tests taxis as a repeatable control loop
	rather than a single lucky approach.

	Note the fitness is an integer count, so it is a coarse, plateau-heavy signal — most genotypes
	score 0 and selection sees no gradient among them. For MAP-Elites that is usually fine (the
	descriptors do the spreading), but if you need a smoother objective, `info` carries the
	distance to the live beacon so you can add partial credit for approach.
	"""
	#-------------------------------------------------------------------
	def __init__(self, agent_interface: AgentInterface, grid_size: tuple[int, int] = (32, 32),
	             body_size: float = 2.0, field: str = "gradient", radius: float | None = None,
	             sigma: float = 5.0, steps: int = 32, n_channels: int = 1, channel: int = 0,
	             reach_threshold: float = 2.0, min_spawn_distance: float | None = None,
	             spawn_tries: int = 8, margin: float = 1.0, beacon_bonus: float = 1.0):
		"""`reach_threshold` is how close counts as reaching a beacon; `min_spawn_distance` is how
		far a replacement must be from the agent (defaults to `radius`, so every leg is about as
		long as the first). Without that floor a beacon could appear on top of the agent and chain
		instantly, inflating the count without any taxis. `spawn_tries` candidates are drawn and
		the first satisfying the floor is used, falling back to the farthest if none do."""
		super().__init__(agent_interface, grid_size, body_size, field, radius, sigma, steps,
		                 n_channels, channel)
		self.reach_threshold = float(reach_threshold)
		self.min_spawn_distance = float(min_spawn_distance) if min_spawn_distance is not None \
			else float(self.radius)
		self.spawn_tries = max(1, int(spawn_tries))
		self.margin = float(margin)
		self.beacon_bonus = float(beacon_bonus)
	#-------------------------------------------------------------------
	def _sample_source(self, agent_pos: jax.Array, key: jax.Array) -> jax.Array:
		"""Draw a replacement beacon at least `min_spawn_distance` from the agent."""
		hi = jnp.asarray(self.grid_size, jnp.float32) - self.margin
		cand = jr.uniform(key, (self.spawn_tries, 2), minval=self.margin,
		                  maxval=hi[None], dtype=jnp.float32)
		d = jnp.linalg.norm(cand - agent_pos.astype(jnp.float32)[None], axis=-1)
		ok = d >= self.min_spawn_distance
		# first candidate clearing the floor; if none do, the farthest one
		idx = jnp.where(ok.any(), jnp.argmax(ok), jnp.argmax(d))
		return cand[idx]
	#-------------------------------------------------------------------
	def reset(self, params: NeuralParams, key: jax.Array, neural_state=None) -> MiniMultiTaxisState:
		base = super().reset(params, key, neural_state=neural_state)
		return MiniMultiTaxisState(state_grid=base.state_grid, agent_state=base.agent_state,
		                           source=base.source, n_reached=jnp.zeros((), jnp.int32),
		                           dist_sum=jnp.zeros((), jnp.float32))
	#-------------------------------------------------------------------
	def step(self, state: MiniMultiTaxisState, key: jax.Array) -> MiniMultiTaxisState:
		k_step, k_beacon = jr.split(key)
		state = super().step(state, k_step)          # move the agent first, then test

		pos = state.agent_state.body.pos.astype(jnp.float32)
		# distance to the beacon that was live *during* this step, accumulated here rather than
		# reconstructed afterwards: the stacked `source` is post-respawn, so measuring after the
		# fact would score the very step the agent succeeded on against the new, distant beacon
		dist = jnp.linalg.norm(pos - state.source.astype(jnp.float32))
		reached = dist < self.reach_threshold

		candidate = self._sample_source(pos, k_beacon)
		source = jnp.where(reached, candidate, state.source)
		# the field only changes when the beacon does, so rebuild it under a cond rather than
		# every step: chem_field is a full [C, H, W] construction
		state_grid = jax.lax.cond(reached, lambda s: self.chem_field(s),
		                          lambda s: state.state_grid, source)

		return state.replace(state_grid=state_grid, source=source,
		                     n_reached=state.n_reached + reached.astype(jnp.int32),
		                     dist_sum=state.dist_sum + dist)
	#-------------------------------------------------------------------
	def evaluate(self, params: NeuralParams, key: jax.Array) -> tuple[Float, dict]:
		"""Fitness = beacons reached, plus continuous credit for staying close to the live one.

		    fitness = beacon_bonus * n_reached + (1 - mean_distance / arena_diagonal)

		The proximity term is the running distance sum divided by `steps * diagonal`, so it lands
		in [0, 1] whatever the rollout length or arena size. `beacon_bonus >= 1` therefore makes
		one arrival worth more than the entire span of the proximity term — an agent that reaches
		a beacon always beats one that merely loiters near it, while genotypes that reach nothing
		are still ranked by how close they got.

		The count alone gave no such ranking: it is an integer that is 0 for nearly every genotype
		in a random population, so selection saw a flat landscape exactly where it needs to climb.
		"""
		states: MiniMultiTaxisState = self.rollout(params, self.steps, key)
		n_reached = states.n_reached[-1]
		dist_sum = states.dist_sum[-1]

		diag = float(np.linalg.norm(np.asarray(self.grid_size, np.float64)))
		mean_norm = dist_sum / (self.steps * diag)          # in [0, 1]
		proximity = 1.0 - jnp.clip(mean_norm, 0.0, 1.0)
		fitness = self.beacon_bonus * n_reached.astype(jnp.float32) + proximity

		pos = states.agent_state.body.pos.astype(jnp.float32)
		path = jnp.linalg.norm(jnp.diff(pos, axis=0), axis=-1).sum()

		return fitness, {
			"n_reached": n_reached,
			"proximity": proximity,
			"mean_distance": dist_sum / self.steps,
			"d_to_live_beacon": jnp.linalg.norm(pos[-1] - states.source[-1].astype(jnp.float32)),
			"path_length": path,
		}
