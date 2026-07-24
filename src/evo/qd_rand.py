r"""MAP-Elites over the RAND developmental encoding.

A genotype here is the RAND GRN parameter pytree (`eqx.filter(RAND_CTRNN(...), eqx.is_array)`).
Evaluating one means *growing* it — running `dev_iters` of regulation/migration/mitosis —
and then reading the resulting CTRNN:

	genotype --develop--> RANDCTRNNState --> descriptors (what the network *is*)
	                                     \-> fitness     (what the network *does*)

Descriptors are structural properties of the grown network (neuron count, wiring density,
spatial spread, ...), so the archive maps out which morphologies the encoding can reach.
The default fitness is `dynamical_richness`: how much sustained activity the grown CTRNN
produces under random drive, which separates dead/saturated networks from live ones.

Fitness and descriptors are the plug-in points: pick from the `fitness_fns` /
`descriptor_fns` registries by name, pass your own callable to `make_rand_scoring_fn`,
or replace `scoring_fn` wholesale with a task-based one (e.g. a gridworld rollout) and
keep the same archive machinery.
"""

import inspect

import jax
from jax import numpy as jnp, random as jr
from jax.flatten_util import ravel_pytree
from jax.sharding import PartitionSpec, NamedSharding
import equinox as eqx
from flax.struct import PyTreeNode
from jaxtyping import PyTree
from typing import Callable

from .qd import MapElitesRepertoire, GeneticAlgorithmRepertoire
from ..devo.core import Body, Observation
from ..devo.motor import BraitenbergMotorInterface, CiliatedMotorInterface, MotorInterface
from ..devo.nn import make_apply_init
from ..devo.nn.rand import RAND_CTRNN, RANDCTRNNState
from ..devo.sensory import RetinaSensoryInterface, SensoryInterface
from ..eco.mini import MiniTaxis, MiniMultiTaxis

type Genotype = PyTree
type Repertoire = MapElitesRepertoire | GeneticAlgorithmRepertoire

# ======================================================================
# descriptors
# ======================================================================

def nb_neurons(net: RANDCTRNNState, model: RAND_CTRNN, key: jax.Array | None = None) -> jax.Array:
	"""Fraction of the neuron budget the embryo actually grew into."""
	return net.mask.sum() / model.max_neurons

def connectivity(net: RANDCTRNNState, model: RAND_CTRNN, key: jax.Array | None = None) -> jax.Array:
	"""Density of realised synapses among living neurons."""
	pairs = net.mask[:, None] & net.mask[None, :]
	n = pairs.sum()
	return jnp.where(n > 0, ((jnp.abs(net.W) > 0) & pairs).sum() / jnp.clip(n, 1), 0.0)

def spatial_extent(net: RANDCTRNNState, model: RAND_CTRNN, key: jax.Array | None = None) -> jax.Array:
	"""Mean distance of living neurons to their centroid (positions live in [-1,1]^2)."""
	mask = net.mask
	n = mask.sum()
	centroid = (net.x * mask[:, None]).sum(0) / jnp.clip(n, 1)
	dists = jnp.linalg.norm(net.x - centroid, axis=-1)
	return jnp.where(n > 0, (dists * mask).sum() / jnp.clip(n, 1), 0.0)

def sensory_expression(net: RANDCTRNNState, model: RAND_CTRNN, key: jax.Array | None = None) -> jax.Array:
	"""Mean sensory-gene expression over living neurons."""
	mask = net.mask
	n = mask.sum()
	return jnp.where(n > 0, (net.s.mean(-1) * mask).sum() / jnp.clip(n, 1), 0.0)

def motor_expression(net: RANDCTRNNState, model: RAND_CTRNN, key: jax.Array | None = None) -> jax.Array:
	"""Mean motor-gene expression over living neurons."""
	mask = net.mask
	n = mask.sum()
	return jnp.where(n > 0, (net.m.mean(-1) * mask).sum() / jnp.clip(n, 1), 0.0)

def excitation_ratio(net: RANDCTRNNState, model: RAND_CTRNN, key: jax.Array | None = None) -> jax.Array:
	"""Share of excitatory (positive) synapses among the realised ones."""
	pairs = net.mask[:, None] & net.mask[None, :]
	realised = (jnp.abs(net.W) > 0) & pairs
	n = realised.sum()
	return jnp.where(n > 0, ((net.W > 0) & realised).sum() / jnp.clip(n, 1), 0.0)

def random_descriptor(net, model, key):
	return jr.uniform(key)


def _sensory_motor_masks(net: RANDCTRNNState, threshold: float) -> tuple[jax.Array, jax.Array]:
	"""(is_sensory, is_motor) boolean masks over living neurons, by gene expression.

	Gene axes are averaged (RAND grows `s`/`m` as [N, genes]) to one scalar per neuron,
	matching how the interfaces reduce them, then thresholded.
	"""
	mask = net.mask
	s = net.s.mean(-1) if net.s.ndim == 2 else net.s
	m = net.m.mean(-1) if net.m.ndim == 2 else net.m
	return (s > threshold) & mask, (m > threshold) & mask


def make_neuron_type_count(kind: str, threshold: float = 0.1) -> Callable:
	"""Descriptor: fraction of the neuron budget that is `kind` (sensory / motor / sensorimotor).

	The three kinds partition the neurons that express a sensory and/or motor gene above
	`threshold` into **disjoint** groups:
		"sensory"      — expresses the sensory gene only
		"motor"        — expresses the motor gene only
		"sensorimotor" — expresses both (a neuron that both senses and drives)
	Each returns count / `model.max_neurons`, so bounds are [0, 1] like `nb_neurons`. This is
	expression-based, so it is interface-agnostic (it does not require a neuron to sit on the
	border to count as a motor, unlike whether the ciliated interface actually uses it).
	"""
	assert kind in ("sensory", "motor", "sensorimotor"), \
		f"kind must be 'sensory', 'motor' or 'sensorimotor', got {kind!r}"

	def neuron_type_count(net: RANDCTRNNState, model: RAND_CTRNN, key: jax.Array | None = None) -> jax.Array:
		is_sensory, is_motor = _sensory_motor_masks(net, threshold)
		if kind == "sensory":
			sel = is_sensory & ~is_motor
		elif kind == "motor":
			sel = is_motor & ~is_sensory
		else:
			sel = is_sensory & is_motor
		return sel.sum() / model.max_neurons

	return neuron_type_count


def make_grid_coverage(resolution: int = 8) -> Callable:
	"""Descriptor: fraction of a `resolution` x `resolution` grid holding >=1 neuron.

	Network space [-1, 1]^2 is discretised into `resolution^2` cells (8x8 mirrors an MNIST
	image), and this returns (occupied cells) / (total cells). It measures how much of the
	sensory field the body physically spans — unlike `spatial_extent` (mean spread from the
	centroid) or `nb_neurons` (count), a few neurons flung to distinct cells cover more than
	many neurons piled into one. It caps at min(nb_neurons, resolution^2) / resolution^2.

	Cells are indexed with the retina's convention (`x[:,0]` left/right, `x[:,1]`
	back/front), so a neuron's cell here matches the pixel it reads through
	`RetinaSensoryInterface`.
	"""
	def grid_coverage(net: RANDCTRNNState, model: RAND_CTRNN, key: jax.Array | None = None) -> jax.Array:
		mask = net.mask
		# [-1, 1] -> [0, 1] -> cell index, clipping neurons sitting on the upper bound
		cells = jnp.clip(jnp.floor((net.x + 1.0) / 2.0 * resolution).astype(jnp.int32),
		                 0, resolution - 1)
		flat = cells[:, 1] * resolution + cells[:, 0]  # row-major, one id per neuron
		occupied = jnp.zeros(resolution * resolution, dtype=jnp.bool)
		occupied = occupied.at[flat].max(mask)          # cell is occupied if any live neuron lands in it
		return occupied.sum() / (resolution * resolution)
	return grid_coverage


# name -> (fn, lower bound, upper bound). Bounds define the archive grid.
# A descriptor has signature `f(net, model, key) -> scalar`. The built-ins are deterministic
# and ignore `key` (it defaults to None), but the argument lets you register stochastic
# descriptors (subsampling, noisy probes, ...); the scoring functions pass a fresh key each eval.
descriptor_fns: dict[str, tuple[Callable, float, float]] = {
	"nb_neurons": (nb_neurons, 0.0, 1.0),
	"connectivity": (connectivity, 0.0, 1.0),
	"spatial_extent": (spatial_extent, 0.0, 1.0),
	"sensory_expression": (sensory_expression, 0.0, 1.0),
	"motor_expression": (motor_expression, 0.0, 1.0),
	"excitation_ratio": (excitation_ratio, 0.0, 1.0),
	"grid_coverage": (make_grid_coverage(8), 0.0, 1.0),
	"nb_sensory": (make_neuron_type_count("sensory"), 0.0, 1.0),
	"nb_motor": (make_neuron_type_count("motor"), 0.0, 1.0),
	"nb_sensorimotor": (make_neuron_type_count("sensorimotor"), 0.0, 1.0),
	"random": (random_descriptor, 0.0, 1.0)
}


def descriptor_bounds(names: list[str]) -> tuple[list[float], list[float]]:
	"""Grid bounds for the named descriptors, in order."""
	for name in names:
		assert name in descriptor_fns, f"unknown descriptor {name!r}, pick from {list(descriptor_fns)}"
	mins = [descriptor_fns[n][1] for n in names]
	maxs = [descriptor_fns[n][2] for n in names]
	return mins, maxs


def gene_expression_entropy(state, mask: jax.Array | None = None, bins: int = 10,
                            bounds: tuple[float, float] = (0.0, 1.0), base: float = 2.0,
                            normalize: bool = False) -> jax.Array:
	"""Per-gene entropy of the expression profile across neurons.

	The expression range is cut into `bins` equal bins; for each gene the living neurons
	are histogrammed over those bins and the entropy of that distribution is returned.
	Low entropy means every neuron expresses the gene at much the same level (an
	undifferentiated population); high entropy means the gene splits neurons across many
	expression levels, i.e. it differentiates them.

	Args:
		state: a `RANDDevelopmentalState` (or anything exposing `.s`, and optionally
			`.mask`), or a raw [N, G] expression array. Note the *full* gene vector lives
			on the developmental state — a grown `RANDCTRNNState.s` holds only the sensory
			genes.
		mask: [N] bool over living neurons. Taken from `state.mask` when available;
			defaults to all neurons.
		bins: number of bins the expression range is split into.
		bounds: expression range, matching the model's `expression_bounds`.
		base: log base for the entropy — 2 gives bits.
		normalize: divide by log(bins), the maximum possible entropy, giving [0, 1].

	Returns:
		[G] per-gene entropy. All-zero if no neuron is alive.
	"""
	idx, mask = _discretize_expression(state, mask, bins, bounds)

	counts = (jax.nn.one_hot(idx, bins) * mask[:, None, None]).sum(0)  # [G, bins]
	n_alive = mask.sum()
	p = counts / jnp.clip(n_alive, 1)

	# clip inside the log only: p=0 bins contribute 0*log(eps) = 0, without producing NaN
	entropy = -(p * jnp.log(jnp.clip(p, 1e-12))).sum(-1) / jnp.log(base)
	if normalize:
		entropy = entropy / (jnp.log(bins) / jnp.log(base))
	return jnp.where(n_alive > 0, entropy, 0.0)


def _discretize_expression(state, mask, bins, bounds):
	"""Shared front-end for the entropy measures: -> (binned expression [N, G], mask [N])."""
	s = getattr(state, "s", state)
	if mask is None:
		mask = getattr(state, "mask", None)

	s = jnp.asarray(s)
	assert s.ndim == 2, f"expected expression of shape [N, G], got {s.shape}"
	assert bins > 0, f"bins must be positive, got {bins}"

	n_neurons = s.shape[0]
	mask = jnp.ones(n_neurons, dtype=bool) if mask is None else jnp.asarray(mask).astype(bool)
	assert mask.shape == (n_neurons,), f"expected mask of shape {(n_neurons,)}, got {mask.shape}"

	lo, hi = bounds
	idx = jnp.floor((s - lo) / max(hi - lo, 1e-8) * bins).astype(jnp.int32)
	return jnp.clip(idx, 0, bins - 1), mask


def global_gene_expression_entropy(state, mask: jax.Array | None = None, bins: int = 10,
                                   bounds: tuple[float, float] = (0.0, 1.0), base: float = 2.0,
                                   normalize: bool = False) -> jax.Array:
	"""Joint entropy over whole expression profiles: how many distinct cell types developed.

	Each living neuron's binned expression vector is treated as one symbol, and this is the
	entropy of the distribution over those symbols. It is the *global* counterpart to
	`gene_expression_entropy`, which scores each gene on its own: this one couples the genes,
	so neurons differing in any single gene count as distinct types.

	0 means every neuron shares one profile (no differentiation); the maximum is
	log(n_alive), reached when every neuron is its own type. Note that it is bounded by the
	neuron count, not by `bins**G` — with N neurons you can never observe more than N types.

	For a per-gene average instead, use `gene_expression_entropy(...).mean()`.

	Args:
		state, mask, bins, bounds, base: as in `gene_expression_entropy`.
		normalize: divide by log(n_alive), the maximum attainable here, giving [0, 1].

	Returns:
		Scalar entropy. 0 if fewer than two neurons are alive.
	"""
	idx, mask = _discretize_expression(state, mask, bins, bounds)

	# multiplicity of each neuron's profile among the living
	same_profile = jnp.all(idx[:, None, :] == idx[None, :, :], axis=-1)  # [N, N]
	counts = (same_profile & mask[None, :]).sum(-1)                      # [N]
	n_alive = mask.sum()

	# H = -(1/n) * sum_i log(p_i) over living i, where p_i is the frequency of i's profile.
	# Summing per-neuron rather than per-distinct-profile avoids a dynamic-shaped unique().
	p = counts / jnp.clip(n_alive, 1)
	log_p = jnp.log(jnp.clip(p, 1e-12))  # living neurons always have counts >= 1, so p > 0
	entropy = -jnp.where(mask, log_p, 0.0).sum() / jnp.clip(n_alive, 1) / jnp.log(base)

	if normalize:
		# a single living neuron can only be one type -> max entropy is log(1) = 0
		max_entropy = jnp.log(jnp.clip(n_alive, 2)) / jnp.log(base)
		entropy = entropy / max_entropy
	return jnp.where(n_alive > 1, entropy, 0.0)


# ======================================================================
# fitness
# ======================================================================
#
# A fitness function has signature `f(net, rollout, key) -> scalar`:
#   net     -- the grown RANDCTRNNState
#   rollout -- `rollout(key) -> [activity_steps, max_neurons]`, the trace of `v` obtained
#              by driving the grown CTRNN with random input. Structural fitnesses ignore it.
#   key     -- PRNG key
#
# `make_rand_scoring_fn` also takes a plain `f(net) -> scalar` callable and adapts it,
# so a custom fitness that only reads the final network needs no boilerplate.

def _masked_mean(values: jax.Array, mask: jax.Array) -> jax.Array:
	"""Mean of `values` over living neurons, 0 when none are alive."""
	n = mask.sum()
	return jnp.where(n > 0, (values * mask).sum() / jnp.clip(n, 1), 0.0)


def dynamical_richness(net: RANDCTRNNState, rollout: Callable, key: jax.Array) -> jax.Array:
	"""Mean temporal std of neural activity under random drive.

	Networks that fall silent or saturate at the clipping bound hold a near-constant
	`v` and score ~0; only networks sustaining varied activity score high. The first
	half of the rollout is dropped as transient.
	"""
	vs = rollout(key)
	vs = vs[vs.shape[0] // 2:]
	return _masked_mean(vs.std(0), net.mask)


def mean_activity(net: RANDCTRNNState, rollout: Callable, key: jax.Array) -> jax.Array:
	"""Mean |v| over the post-transient rollout."""
	vs = rollout(key)
	vs = vs[vs.shape[0] // 2:]
	return _masked_mean(jnp.abs(vs).mean(0), net.mask)


def activity_spread(net: RANDCTRNNState, rollout: Callable, key: jax.Array) -> jax.Array:
	"""Std *across* neurons of their mean activity: rewards differentiated neurons.

	High when the network settles into distinct roles rather than every neuron doing
	the same thing.
	"""
	vs = rollout(key)
	vs = vs[vs.shape[0] // 2:]
	per_neuron = vs.mean(0)
	mask = net.mask
	mu = _masked_mean(per_neuron, mask)
	return jnp.sqrt(_masked_mean(jnp.square(per_neuron - mu), mask))


def neuron_count(net: RANDCTRNNState, rollout: Callable, key: jax.Array) -> jax.Array:
	"""Structural: reward growing more neurons."""
	return net.mask.sum().astype(jnp.float32)


def constant(net: RANDCTRNNState, rollout: Callable, key: jax.Array) -> jax.Array:
	"""No selection pressure: the run becomes pure archive-filling / morphospace mapping."""
	return jnp.zeros(())


fitness_fns: dict[str, Callable] = {
	"dynamical_richness": dynamical_richness,
	"mean_activity": mean_activity,
	"activity_spread": activity_spread,
	"neuron_count": neuron_count,
	"constant": constant,
}


def resolve_fitness(fitness: str | Callable) -> Callable:
	"""Normalise a registry name or user callable to the `f(net, rollout, key)` form.

	A callable of arity 1 is treated as `f(net)` and wrapped; anything else is assumed
	to already take `(net, rollout, key)`.
	"""
	if isinstance(fitness, str):
		assert fitness in fitness_fns, f"unknown fitness {fitness!r}, pick from {list(fitness_fns)}"
		return fitness_fns[fitness]
	assert callable(fitness), f"fitness must be a name or a callable, got {type(fitness)}"
	try:
		arity = len(inspect.signature(fitness).parameters)
	except (TypeError, ValueError):  # builtins / C callables expose no signature
		arity = 1
	if arity == 1:
		return lambda net, rollout, key: fitness(net)
	return fitness


# ======================================================================
# scoring
# ======================================================================

def make_rand_scoring_fn(model: RAND_CTRNN,
                         descriptors: list[str],
                         fitness: str | Callable = "dynamical_richness",
                         activity_steps: int = 100,
                         input_scale: float = 0.1) -> Callable:
	"""Build `eval(genotype, key) -> (fitness, descriptors, extra_scores)` for one genotype.

	Args:
		model: template RAND_CTRNN; only its static structure is used, the genotype
			supplies the parameters.
		descriptors: names from `descriptor_fns`, in archive-axis order.
		fitness: a name from `fitness_fns`, or a callable. A one-argument callable is
			handed the grown network (`f(net) -> scalar`); a callable of any other arity
			gets the full `f(net, rollout, key)` form described above.
		activity_steps, input_scale: length and strength of the random drive used by the
			dynamical fitnesses. Structural fitnesses ignore them.

	The returned function is unbatched; `make_train_fn` vmaps it.
	"""
	apply_fn, init_fn = make_apply_init(model)
	desc_fns = [descriptor_fns[name][0] for name in descriptors]
	fitness_fn = resolve_fitness(fitness)

	def eval_fn(genotype: Genotype, key: jax.Array):
		key_dev, key_fit, key_desc = jr.split(key, 3)
		net = init_fn(genotype, key_dev)  # grow the network

		def rollout(key: jax.Array) -> jax.Array:
			def _step(state, k):
				k_in, k_net = jr.split(k)
				x = jr.normal(k_in, (state.v.shape[0],)) * input_scale
				state, _ = apply_fn(genotype, x, state, k_net)
				return state, state.v
			_, vs = jax.lax.scan(_step, net, jr.split(key, activity_steps))
			return vs  # [activity_steps, max_neurons]

		# cast: the repertoire compares fitness against -inf, so it must be floating
		f = jnp.asarray(fitness_fn(net, rollout, key_fit), dtype=jnp.float32)
		bd = jnp.stack([fn(net, model, k) for fn, k in zip(desc_fns, jr.split(key_desc, len(desc_fns)))])
		return f, bd, {"nb_neurons": net.mask.sum().astype(jnp.float32)}

	return eval_fn


# ======================================================================
# embodiment: score the behaviour a network generates, not just its dynamics
# ======================================================================
#
# The grown network is given a body and a motor interface and left to act. What comes out
# is a Trajectory, and *behaviour statistics* over it play the role that the `v`-statistics
# play for the disembodied scorer. A statistic has signature `f(traj, net, model, key) -> scalar`
# and can serve as fitness, as a descriptor, or both -- hence one registry with bounds. The
# built-ins are deterministic and ignore `key`; it is there so stochastic stats can be registered.

class Trajectory(PyTreeNode):
	#-------------------------------------------------------------------
	pos: jax.Array      # [T, 2]  body position
	heading: jax.Array  # [T]     body heading, wrapped to [0, 2pi)
	action: jax.Array   # [T, 2]  motor command (left/right wheel speed)
	v: jax.Array        # [T, max_neurons]  neural activity
	#-------------------------------------------------------------------


def _embodiable(net: RANDCTRNNState) -> RANDCTRNNState:
	"""Reshape a grown RAND network so the motor interfaces can consume it.

	RAND grows motor expression as `m` of shape [N, motor_genes], but the spatially
	embedded interfaces index it as [N] (e.g. `_init_se` computes `(m > thr) & mask`,
	which would broadcast [N, 1] against [N] into an [N, N] array). Collapse the gene
	axis so each neuron carries a single motor expression.
	"""
	m = net.m
	if m.ndim == 2:
		m = m.mean(-1)
	return net.replace(m=m)


def _wrapped_heading_deltas(heading: jax.Array) -> jax.Array:
	"""Per-step heading change in [-pi, pi]; `move` wraps heading, so raw diffs jump 2pi."""
	dh = jnp.diff(heading.astype(jnp.float32))
	return (dh + jnp.pi) % (2 * jnp.pi) - jnp.pi


def _step_lengths(traj: Trajectory) -> jax.Array:
	return jnp.linalg.norm(jnp.diff(traj.pos.astype(jnp.float32), axis=0), axis=-1)


def distance_travelled(traj: Trajectory, net, model, key: jax.Array | None = None) -> jax.Array:
	"""Total path length walked."""
	return _step_lengths(traj).sum()


def net_displacement(traj: Trajectory, net, model, key: jax.Array | None = None) -> jax.Array:
	"""Straight-line distance from start to finish."""
	pos = traj.pos.astype(jnp.float32)
	return jnp.linalg.norm(pos[-1] - pos[0])


def path_efficiency(traj: Trajectory, net, model, key: jax.Array | None = None) -> jax.Array:
	"""Displacement / path length: 1 = straight line, ~0 = circling or wandering."""
	travelled = distance_travelled(traj, net, model)
	return jnp.where(travelled > 1e-6, net_displacement(traj, net, model) / jnp.clip(travelled, 1e-6), 0.0)


def mean_speed(traj: Trajectory, net, model, key: jax.Array | None = None) -> jax.Array:
	"""Mean distance covered per step."""
	return _step_lengths(traj).mean()


def turn_bias(traj: Trajectory, net, model, key: jax.Array | None = None) -> jax.Array:
	"""Mean signed turn: chirality. >0 turns left on average, <0 right."""
	return _wrapped_heading_deltas(traj.heading).mean()


def mean_abs_turn(traj: Trajectory, net, model, key: jax.Array | None = None) -> jax.Array:
	"""Mean unsigned turn: how much the agent rotates regardless of direction."""
	return jnp.abs(_wrapped_heading_deltas(traj.heading)).mean()


def motor_activity(traj: Trajectory, net, model, key: jax.Array | None = None) -> jax.Array:
	"""Mean magnitude of the motor command."""
	return jnp.abs(traj.action.astype(jnp.float32)).mean()


# name -> (fn, lower bound, upper bound). Bounds are *defaults* for `behaviour_bounds`:
# the distance-like ones scale with `steps` and wheel speed, so override them to match
# your rollout rather than trusting these.
behaviour_fns: dict[str, tuple[Callable, float, float]] = {
	"distance_travelled": (distance_travelled, 0.0, 20.0),
	"net_displacement": (net_displacement, 0.0, 20.0),
	"path_efficiency": (path_efficiency, 0.0, 1.0),
	"mean_speed": (mean_speed, 0.0, 0.5),
	"turn_bias": (turn_bias, -jnp.pi, jnp.pi),
	"mean_abs_turn": (mean_abs_turn, 0.0, jnp.pi),
	"motor_activity": (motor_activity, 0.0, 1.0),
}


def resolve_behaviour_stat(name: str) -> tuple[Callable, float, float]:
	"""Look a statistic up in `behaviour_fns`, falling back to the structural
	`descriptor_fns` (adapted to the `f(traj, net, model, key)` signature)."""
	if name in behaviour_fns:
		return behaviour_fns[name]
	if name in descriptor_fns:
		fn, lo, hi = descriptor_fns[name]
		return (lambda traj, net, model, key=None, _fn=fn: _fn(net, model, key)), lo, hi
	raise AssertionError(
		f"unknown statistic {name!r}, pick from {list(behaviour_fns)} or {list(descriptor_fns)}"
	)


def behaviour_bounds(names: list[str]) -> tuple[list[float], list[float]]:
	"""Default grid bounds for the named statistics, in order."""
	resolved = [resolve_behaviour_stat(n) for n in names]
	return [lo for _, lo, _ in resolved], [hi for _, _, hi in resolved]


def make_rand_embodied_scoring_fn(model: RAND_CTRNN,
                                  descriptors: list[str],
                                  fitness: str | Callable = "distance_travelled",
                                  motor_interface: MotorInterface | None = None,
                                  sensory_interface: SensoryInterface | None = None,
                                  obs_fn: Callable | None = None,
                                  steps: int = 100,
                                  body_size: float = 2.0,
                                  init_heading: float = 0.0) -> Callable:
	"""Build `eval(genotype, key) -> (fitness, descriptors, extra)` scoring *behaviour*.

	The genotype is grown, the network is put in a body, and the motor interface turns its
	activity into movement for `steps` ticks. Fitness and descriptors are then read off the
	resulting `Trajectory` rather than off raw `v` statistics.

	Args:
		model: template RAND_CTRNN.
		descriptors: names from `behaviour_fns` or `descriptor_fns` (structural and
			behavioural descriptors can be mixed), in archive-axis order.
		fitness: a name from those same registries, or a callable. A one-argument callable
			is handed the trajectory (`f(traj) -> scalar`); any other arity gets the full
			`f(traj, net, model)` form.
		motor_interface: defaults to `BraitenbergMotorInterface(interface="se")`, which
			reads the grown neurons' positions and motor expression.
		sensory_interface: optional. If given, `obs_fn` must be given too.
		obs_fn: `obs_fn(body, key) -> Observation`, the seam for an environment. When None
			the agent is open-loop: the network receives zero input and behaviour comes from
			its intrinsic dynamics alone. Note the `spatially_embedded` sensory interface
			needs `obs.env` of shape [C, body_resolution, body_resolution] and requires the
			model's `sensory_genes` to equal C + len(obs.internal).
		steps: rollout length.
		body_size: body diameter; the Braitenberg wheelbase is derived from it.
		init_heading: starting heading, in radians.

	The returned function is unbatched; `make_train_fn` vmaps it.
	"""
	assert (sensory_interface is None) == (obs_fn is None), \
		"pass sensory_interface and obs_fn together, or neither (open-loop)"

	motor_interface = motor_interface if motor_interface is not None \
		else BraitenbergMotorInterface(interface="se")

	apply_fn, init_fn = make_apply_init(model)
	stat_fns = [resolve_behaviour_stat(name)[0] for name in descriptors]

	if isinstance(fitness, str):
		fitness_fn = resolve_behaviour_stat(fitness)[0]
	else:
		assert callable(fitness), f"fitness must be a name or a callable, got {type(fitness)}"
		try:
			arity = len(inspect.signature(fitness).parameters)
		except (TypeError, ValueError):
			arity = 1
		fitness_fn = (lambda traj, net, model, _f=fitness: _f(traj)) if arity == 1 else fitness

	# bodies are float16 throughout the simulator, and the motor interfaces emit float16
	# actions; keeping the body in float16 avoids a dtype mismatch inside `move`
	dtype = jnp.float16

	def rollout(genotype: Genotype, net: RANDCTRNNState, key: jax.Array) -> Trajectory:
		key_motor, key_sensory, key_scan = jr.split(key, 3)
		motor_state = motor_interface.init(net, key=key_motor)
		sensory_state = None if sensory_interface is None else sensory_interface.init(net, key_sensory)
		body = Body(pos=jnp.zeros(2, dtype=dtype),
		            heading=jnp.astype(init_heading, dtype),
		            size=jnp.astype(body_size, dtype))

		def _step(carry, k):
			net, body, sensory_state, motor_state = carry
			if sensory_interface is None:
				I = jnp.zeros((net.v.shape[0],))
			else:
				k_obs, k = jr.split(k)
				obs = obs_fn(body, k_obs)  #type:ignore[misc]
				I, _, sensory_state, _ = sensory_interface.encode(obs, net, sensory_state)
			k_apply, k_act = jr.split(k)
			net, _ = apply_fn(genotype, I, net, k_apply)
			body, _, motor_state, motor_info = motor_interface.actuate(net, motor_state, body, k_act)
			return (net, body, sensory_state, motor_state), (body.pos, body.heading, motor_info["action"], net.v)

		_, (pos, heading, action, v) = jax.lax.scan(
			_step, (net, body, sensory_state, motor_state), jr.split(key_scan, steps)
		)
		return Trajectory(pos=pos, heading=heading, action=action, v=v)

	def eval_fn(genotype: Genotype, key: jax.Array):
		key_dev, key_roll, key_desc = jr.split(key, 3)
		net = _embodiable(init_fn(genotype, key_dev))  # grow, then wire to the body
		traj = rollout(genotype, net, key_roll)

		f = jnp.asarray(fitness_fn(traj, net, model), dtype=jnp.float32)
		bd = jnp.stack([jnp.asarray(fn(traj, net, model, k), dtype=jnp.float32)
		                for fn, k in zip(stat_fns, jr.split(key_desc, len(stat_fns)))])
		return f, bd, {"nb_neurons": net.mask.sum().astype(jnp.float32)}

	return eval_fn


# ======================================================================
# developmental plasticity: same genotype, two developmental conditions
# ======================================================================
#
# A genotype is grown twice, differing only in the initial expression of the last gene. The
# archive's two axes are the *same* descriptor measured in each condition, so the diagonal
# holds canalised genotypes (the condition changed nothing) and the further a cell sits off
# it, the more plastic the genotype. Development only -- nothing is rolled out afterwards.

def plasticity_bounds(descriptor: str) -> tuple[list[float], list[float]]:
	"""Archive bounds for a plasticity run: the same descriptor's bounds on both axes."""
	assert descriptor in descriptor_fns, \
		f"unknown descriptor {descriptor!r}, pick from {list(descriptor_fns)}"
	_, lo, hi = descriptor_fns[descriptor]
	return [lo, lo], [hi, hi]


def make_rand_plasticity_scoring_fn(model: RAND_CTRNN,
                                    descriptor: str = "nb_neurons",
                                    values: tuple[float, float] = (0.0, 1.0),
                                    fitness: str | Callable = "constant") -> Callable:
	"""Build `eval(genotype, key) -> (fitness, descriptors, extra)` scoring developmental plasticity.

	The genotype is developed twice. The runs are identical except that the **last gene**
	(`s[:, -1]`) is clamped to `values[0]` at birth in condition A and to `values[1]` in
	condition B. Both share the same PRNG key, so any difference between the two outcomes is
	caused by that gene and not by developmental noise. The clamp is an initial condition
	only — the GRN evolves it from there.

	Args:
		model: template RAND_CTRNN.
		descriptor: a name from `descriptor_fns`. Measured in both conditions; the two
			results become the two archive axes — build centroids with
			`plasticity_bounds(descriptor)`.
		values: the last gene's initial expression in condition A and condition B.
		fitness: "constant" (default) applies no selection pressure, so the run just maps out
			the plasticity plane; "plasticity" scores |descriptor_A - descriptor_B|. A
			callable gets `f(net_a, net_b, model)`, or `f(net_a, net_b)` if it takes two
			arguments.

	The returned function is unbatched; `make_train_fn` vmaps it.
	"""
	assert descriptor in descriptor_fns, \
		f"unknown descriptor {descriptor!r}, pick from {list(descriptor_fns)}"
	desc_fn = descriptor_fns[descriptor][0]

	# the genotype supplies the arrays; the template carries the static structure
	_, static = eqx.partition(model, eqx.is_array)

	def develop(genotype: Genotype, key: jax.Array, value: float) -> RANDCTRNNState:
		mdl = eqx.combine(genotype, static)
		key_embryo, key_dev = jr.split(key)
		embryo = mdl.init_embryo(key_embryo)
		embryo = embryo.replace(s=embryo.s.at[:, -1].set(value))  # clamp the last gene at birth
		grown, _ = mdl.do_migration(embryo, key_dev)
		return mdl.make_network(grown)

	if isinstance(fitness, str):
		assert fitness in fitness_fns.keys(), \
			f"unknown fitness {fitness!r}, use 'constant', 'plasticity' or a callable"
		fitness_fn = fitness_fns[fitness]
	else:
		assert callable(fitness), f"fitness must be a name or a callable, got {type(fitness)}"
		fitness_kind = "callable"
		try:
			arity = len(inspect.signature(fitness).parameters)
		except (TypeError, ValueError):
			arity = 3
		fitness_fn = (lambda a, b, m, _f=fitness: _f(a, b)) if arity == 2 else fitness

	def eval_fn(genotype: Genotype, key: jax.Array):
		key_dev, key_fit = jr.split(key)
		# same key for both: the only difference between the runs is the clamped gene
		net_a = develop(genotype, key_dev, values[0])
		net_b = develop(genotype, key_dev, values[1])

		desc_a = jnp.asarray(desc_fn(net_a, model), dtype=jnp.float32)
		desc_b = jnp.asarray(desc_fn(net_b, model), dtype=jnp.float32)
		bd = jnp.stack([desc_a, desc_b])

		f = 0.5*(fitness_fn(net_a, None,  key_fit) + fitness_fn(net_b, None,  key_fit))

		return f, bd, {"nb_neurons_a": net_a.mask.sum().astype(jnp.float32),
		               "nb_neurons_b": net_b.mask.sum().astype(jnp.float32)}

	return eval_fn


# ======================================================================
# classification: fitness = accuracy on a tiny digit task
# ======================================================================
#
# The grown network is used as a retina: each image is stretched over network space and every
# sensory neuron reads the pixel under it. The image is held on for `present_steps` ticks so
# the CTRNN can integrate it, then the motor genes act as class readouts (gene k scores class
# k) and the argmax is the prediction. Fitness is accuracy; descriptors stay structural.

def load_small_mnist(classes: tuple[int, ...] | None = (0, 1), n_samples: int = 64,
                     seed: int = 0, normalize: bool = True) -> tuple[jax.Array, jax.Array]:
	"""A balanced digit set from sklearn's bundled 8x8 digits.

	This is `sklearn.datasets.load_digits` — 1797 images at 8x8 with values 0..16, and no
	download needed. It is genuinely tiny, which suits a retina whose resolution is set by
	where neurons happen to grow. Swap in real MNIST by passing your own arrays to
	`make_rand_classification_scoring_fn`.

	Args:
		classes: which digits to tell apart, any number of them. `None` means all ten.
			Labels are remapped to 0..len(classes)-1, in the order given.
		n_samples: total images, split evenly across the classes, so chance accuracy is
			1/len(classes). Must be divisible by the number of classes. The scarcest digit
			has 174 examples, so all ten classes allow at most 1740.
		seed: controls which images are drawn and their order.
		normalize: scale pixels from 0..16 to 0..1.

	Returns:
		(images [n_samples, 8, 8] float32, labels [n_samples] int32 in 0..len(classes)-1).
	"""
	from sklearn.datasets import load_digits
	import numpy as np

	classes = tuple(range(10)) if classes is None else tuple(classes)
	n_classes = len(classes)
	assert n_classes >= 2, f"need at least two classes, got {classes}"
	assert len(set(classes)) == n_classes, f"duplicate classes in {classes}"
	assert n_samples % n_classes == 0, \
		f"n_samples must divide by the {n_classes} classes to stay balanced, got {n_samples}"

	digits = load_digits()
	images, targets = digits.images, digits.target

	keep = np.isin(targets, classes)
	images, targets = images[keep], targets[keep]
	# remap each digit to its index in `classes`
	lookup = {digit: i for i, digit in enumerate(classes)}
	labels = np.array([lookup[t] for t in targets], dtype=np.int32)

	rng = np.random.default_rng(seed)
	per_class = n_samples // n_classes
	picked = []
	for i, digit in enumerate(classes):
		available = np.flatnonzero(labels == i)
		assert len(available) >= per_class, \
			f"only {len(available)} images of digit {digit}, need {per_class}"
		picked.append(rng.choice(available, per_class, replace=False))
	picked = rng.permutation(np.concatenate(picked))

	images, labels = images[picked], labels[picked]
	if normalize:
		images = images / 16.0
	return jnp.asarray(images, dtype=jnp.float32), jnp.asarray(labels, dtype=jnp.int32)


def make_rand_classification_scoring_fn(model: RAND_CTRNN,
                                        images: jax.Array,
                                        labels: jax.Array,
                                        n_classes: int | None = None,
                                        descriptors: list[str] = ["nb_neurons", "connectivity"],
                                        sensory_interface: SensoryInterface | None = None,
                                        present_steps: int = 10) -> Callable:
	"""Build `eval(genotype, key) -> (accuracy, descriptors, extra)` for an image task.

	Each image is presented to the grown network through a retina for `present_steps` ticks,
	starting from the freshly grown state each time (so images do not bleed into each other).
	The network's **motor genes act as class readouts**: gene k scores class k, giving
	`logit[k] = sum_i v[i] * m[i, k] * mask[i]`, and the prediction is the argmax. This needs
	`motor_genes >= n_classes` in the model, so build the model with `motor_genes=n_classes`.
	Fitness is accuracy over all `images`; chance is 1/n_classes on a balanced set.

	A network with no motor expression has all-zero logits and so predicts one class for
	everything, scoring ~chance. Likewise a network with no *sensory* neurons sees nothing.
	Both are the floor this task selects away from.

	Args:
		model: template RAND_CTRNN, built with `motor_genes >= n_classes`.
		images: [n, H, W] grayscale, ideally scaled to 0..1.
		labels: [n] integer class indices in 0..n_classes-1.
		n_classes: number of classes; inferred from `labels` when None, which is only safe
			if every class appears in the batch.
		descriptors: names from `descriptor_fns`; defaults to the neuron count and wiring
			density. Build centroids with `descriptor_bounds(descriptors)`.
		sensory_interface: defaults to `RetinaSensoryInterface` matched to the image size.
		present_steps: ticks the image is held on for before reading out.

	The returned function is unbatched; `make_train_fn` vmaps it.
	"""
	images = jnp.asarray(images)
	labels = jnp.asarray(labels)
	assert images.ndim == 3, f"expected images of shape [n, H, W], got {images.shape}"
	assert labels.shape == (images.shape[0],), \
		f"got {labels.shape} labels for {images.shape[0]} images"

	n_classes = int(labels.max()) + 1 if n_classes is None else n_classes
	assert n_classes >= 2, f"need at least two classes, got {n_classes}"

	# the motor compartment supplies one readout channel per class
	motor_genes = int(jnp.asarray(model.genes_shaper(jnp.zeros(model.total_genes))["motor"]).shape[0])
	assert motor_genes >= n_classes, \
		f"the readout needs one motor gene per class: build the model with " \
		f"motor_genes>={n_classes}, got motor_genes={motor_genes}"

	n_images, height, width = images.shape
	if sensory_interface is None:
		sensory_interface = RetinaSensoryInterface(height=height, width=width)

	apply_fn, init_fn = make_apply_init(model)
	desc_fns = [descriptor_fns[name][0] for name in descriptors]
	no_internal = jnp.zeros(4)  # the retina ignores internal signals, but Observation wants them

	def classify(genotype: Genotype, net: RANDCTRNNState, layout, readout: jax.Array,
	             image: jax.Array, key: jax.Array) -> jax.Array:
		def _step(state, k):
			I, _, _, _ = sensory_interface.encode(Observation(env=image, internal=no_internal),
			                                      state, layout)
			state, _ = apply_fn(genotype, I, state, k)
			return state, None

		# each image starts from the grown state, whose v is zero -> no carry-over
		final, _ = jax.lax.scan(_step, net, jr.split(key, present_steps))
		return (final.v[:, None] * readout).sum(0)  # [n_classes]

	def eval_fn(genotype: Genotype, key: jax.Array):
		key_dev, key_run, key_desc = jr.split(key, 3)
		net = init_fn(genotype, key_dev)
		layout = sensory_interface.init(net, key_dev)

		# one readout weight per (living neuron, class). Kept separate from `net` so the
		# descriptors still see the network as it grew.
		readout = net.m[:, :n_classes] * net.mask[:, None]  # [N, n_classes]

		logits = jax.vmap(classify, in_axes=(None, None, None, None, 0, 0))(
			genotype, net, layout, readout, images, jr.split(key_run, n_images)
		)  # [n_images, n_classes]
		accuracy = (jnp.argmax(logits, axis=-1) == labels).mean().astype(jnp.float32)

		bd = jnp.stack([jnp.asarray(fn(net, model, k), dtype=jnp.float32)
		                for fn, k in zip(desc_fns, jr.split(key_desc, len(desc_fns)))])
		return accuracy, bd, {"nb_neurons": net.mask.sum().astype(jnp.float32),
		                      "nb_sensors": layout.is_sensor.sum().astype(jnp.float32)}

	return eval_fn


# ======================================================================
# chemotaxis: fitness = a MiniTaxis rollout toward a crafted chemical beacon
# ======================================================================
#
# Unlike the other embodied scorers (which roll the network out with a bare motor interface
# and no environment), this one drives the full sensorimotor loop through a `MiniTaxis`
# env: the grown agent senses a static chemical field at its body points, acts, and moves,
# and fitness rewards ending closer to the beacon. It therefore needs a full
# `AgentInterface` (sensory + motor), not just the RAND model.

def make_rand_taxis_scoring_fn(agent_interface,
                               model: RAND_CTRNN,
                               descriptors: list[str] = ["nb_neurons", "connectivity"],
                               grid_size: tuple[int, int] = (32, 32),
                               field: str = "gradient",
                               steps: int = 32,
                               radius: float | None = None,
                               sigma: float = 5.0) -> Callable:
	"""Build `eval(genotype, key) -> (fitness, descriptors, extra)` for the chemotaxis task.

	The genotype is the RAND neural-parameter pytree (as produced by
	`agent_interface.neural_fctry`, i.e. what `make_rand_genotype_fctry` returns). Fitness is
	the `MiniTaxis` taxis score `(d_start - d_end)/d_start` toward a beacon at a random
	bearing (drawn from the eval key), so scoring over several keys — via `nb_evals` in
	`make_train_fn` — rewards general taxis rather than one direction.

	Args:
		agent_interface: a full `AgentInterface` (spatially-embedded sensory + ciliated /
			braitenberg motor) whose neural model is `model`. Grows the agent and runs the
			sensorimotor loop. Config contract: the field is 1 channel and the interface
			prepends 4 internal signals, so `sensory_genes == 5`.
		model: the RAND_CTRNN template, used only to evaluate the structural descriptors.
		descriptors: names from `descriptor_fns`; the archive axes. Default neuron count x
			wiring density. Build centroids with `descriptor_bounds(descriptors)`.
		grid_size, field, steps, radius, sigma: forwarded to `MiniTaxis` (see its docstring;
			`field="gradient"` gives a usable signal everywhere).

	The returned function is unbatched; `make_train_fn` vmaps it.
	"""
	env = MiniTaxis(agent_interface, grid_size=grid_size, field=field,
	                steps=steps, radius=radius, sigma=sigma)
	desc_fns = [descriptor_fns[name][0] for name in descriptors]

	def eval_fn(genotype: Genotype, key: jax.Array):
		key_taxis, key_dev, key_desc = jr.split(key, 3)
		fitness, info = env.evaluate(genotype, key_taxis)
		# descriptors describe an independently grown realization of the same genotype;
		# with stochastic development this differs slightly from the one that was rolled out,
		# which `nb_evals` averaging smooths over.
		net = agent_interface.neural_init(genotype, key_dev)
		bd = jnp.stack([jnp.asarray(fn(net, model, k), dtype=jnp.float32)
		                for fn, k in zip(desc_fns, jr.split(key_desc, len(desc_fns)))])
		extra = {"d_start": info["d_start"], "d_end": info["d_end"],
		         "mean_conc": info["mean_conc"], "nb_neurons": net.mask.sum().astype(jnp.float32)}
		return jnp.asarray(fitness, dtype=jnp.float32), bd, extra

	return eval_fn


def make_rand_multitaxis_scoring_fn(agent_interface,
                                    model: RAND_CTRNN,
                                    descriptors: list[str] = ["nb_neurons", "connectivity"],
                                    grid_size: tuple[int, int] = (64, 64),
                                    field: str = "gradient",
                                    steps: int = 200,
                                    radius: float | None = None,
                                    sigma: float = 5.0,
                                    n_channels: int = 1,
                                    channel: int = 0,
                                    reach_threshold: float = 2.0,
                                    min_spawn_distance: float | None = None,
                                    beacon_bonus: float = 1.0) -> Callable:
	"""Build `eval(genotype, key) -> (fitness, descriptors, extra)` for sequential chemotaxis.

	Same contract as `make_rand_taxis_scoring_fn`, but on `MiniMultiTaxis`: one beacon at a time,
	replaced whenever it is reached. That rewards *repeatable* taxis — a fixed turning bias can
	luck onto a single beacon but cannot chain them, because each replacement sits at a new
	bearing.

	Fitness is `MiniMultiTaxis.evaluate`'s: `beacon_bonus * n_reached + proximity`, where
	proximity in [0, 1] falls off with the mean distance held to the live beacon. See that
	docstring for why the raw count alone is too flat to select on.

	Args:
		agent_interface: full `AgentInterface` whose neural model is `model`. Config contract:
			`sensory_genes == n_channels + 4` (field channels plus the 4 internal signals).
		model: RAND_CTRNN template, used only for the structural descriptors.
		descriptors: names from `descriptor_fns`; the archive axes.
		n_channels / channel: which observation channel carries the beacon (chemicals in `ct-*`
			order, then walls), matching the full simulation's channel layout.
		reach_threshold / min_spawn_distance: forwarded to `MiniMultiTaxis`.

	The returned function is unbatched; `make_train_fn` vmaps it.
	"""
	env = MiniMultiTaxis(agent_interface, grid_size=grid_size, field=field, steps=steps,
	                     radius=radius, sigma=sigma, n_channels=n_channels, channel=channel,
	                     reach_threshold=reach_threshold, min_spawn_distance=min_spawn_distance,
	                     beacon_bonus=beacon_bonus)
	desc_fns = [descriptor_fns[name][0] for name in descriptors]

	def eval_fn(genotype: Genotype, key: jax.Array):
		key_task, key_dev, key_desc = jr.split(key, 3)
		fitness, info = env.evaluate(genotype, key_task)
		# descriptors describe an independently grown realization of the same genotype; with
		# stochastic development this differs slightly from the one that was rolled out, which
		# `nb_evals` averaging smooths over.
		net = agent_interface.neural_init(genotype, key_dev)
		bd = jnp.stack([jnp.asarray(fn(net, model, k), dtype=jnp.float32)
		                for fn, k in zip(desc_fns, jr.split(key_desc, len(desc_fns)))])
		extra = {"n_reached": info["n_reached"].astype(jnp.float32),
		         "proximity": info["proximity"],
		         "mean_distance": info["mean_distance"],
		         "path_length": info["path_length"],
		         "nb_neurons": net.mask.sum().astype(jnp.float32)}
		return jnp.asarray(fitness, dtype=jnp.float32), bd, extra

	return eval_fn


# ======================================================================
# mutation
# ======================================================================

class ContinuousMutation(eqx.Module):
	"""Per-parameter Gaussian mutation over a raw genotype pytree."""
	#-------------------------------------------------------------------
	sigma_mut: float
	mut_rate: float
	#-------------------------------------------------------------------
	def __call__(self, genotype: Genotype, key: jax.Array, genotype_infos: PyTree = None) -> Genotype:
		k_mut, k_loc = jr.split(key)
		flat, shaper = ravel_pytree(genotype)
		mutation = jr.normal(k_mut, flat.shape) * self.sigma_mut
		mutation = jnp.where(jr.bernoulli(k_loc, self.mut_rate, flat.shape), mutation, 0.0)
		return shaper(flat + mutation)
	#-------------------------------------------------------------------


# ======================================================================
# training loop
# ======================================================================

class TrainState(PyTreeNode):
	#-------------------------------------------------------------------
	repertoire: MapElitesRepertoire
	steps: int
	key: jax.Array
	#-------------------------------------------------------------------


def qd_metrics(repertoire: MapElitesRepertoire) -> dict:
	occupied = repertoire.is_occupied()
	n = occupied.sum()
	fitnesses = jnp.where(occupied, repertoire.fitnesses, 0.0)
	return {
		"coverage": n / repertoire.num_centroids,
		"nb_elites": n,
		"qd_score": fitnesses.sum(),
		"max_fitness": jnp.where(occupied, repertoire.fitnesses, -jnp.inf).max(),
		"mean_fitness": jnp.where(n > 0, fitnesses.sum() / jnp.clip(n, 1), 0.0),
	}

def make_train_fn(genotype_fctry: Callable[[jax.Array], Genotype],
                  scoring_fn: Callable,
                  mutation_fn: Callable,
                  centroids: jax.Array,
                  N: int,
                  train_steps: int,
				  repertoire_cls: type[Repertoire]=MapElitesRepertoire,
                  nb_evals: int = 1,
                  nb_init_genotypes: int | None = None,
                  nb_devices: int | None = None,
                  return_trace: bool = True,
                  verbose: bool = False,
                  repertoire_kwargs: dict = {}) -> Callable:
	"""Build the MAP-Elites loop.

	Args:
		genotype_fctry: key -> genotype, used to seed the archive.
		scoring_fn: unbatched (genotype, key) -> (fitness, descriptors, extra_scores).
		mutation_fn: unbatched (genotype, key, genotype_infos) -> genotype.
		centroids: [num_centroids, B] archive centroids.
		N: batch size (offspring per iteration). Must divide evenly across devices.
		train_steps: number of MAP-Elites iterations.
		nb_evals: how many times to score each genotype, on independent keys, averaging the
			fitness and descriptors. With stochastic development (RAND re-samples the
			phenotype each growth) a single evaluation is a noisy draw, so scoring an elite
			and re-growing it later can disagree; averaging over `nb_evals` realizations
			scores the *expected* phenotype and makes the archive reproducible. Costs
			`nb_evals`x the evaluation compute. 1 keeps the old single-shot behaviour.

	Returns:
		`train(key) -> (final_state, metrics_trace)`.
	"""
	assert nb_evals >= 1, f"nb_evals must be >= 1, got {nb_evals}"
	nb_init_genotypes = N if nb_init_genotypes is None else nb_init_genotypes
	nb_devices = jax.device_count() if nb_devices is None else nb_devices

	if nb_evals == 1:
		eval_one = scoring_fn
	else:
		def eval_one(genotype: Genotype, key: jax.Array):
			# score the same genotype on nb_evals independent growth keys, then average.
			# extra_scores are averaged too, so per-realization counts become expectations.
			fitness, descriptors, extra = jax.vmap(scoring_fn, in_axes=(None, 0))(
				genotype, jr.split(key, nb_evals)
			)
			return (fitness.mean(0),
			        descriptors.mean(0),
			        jax.tree.map(lambda x: x.mean(0), extra))

	batched_eval_fn = jax.vmap(eval_one)
	batched_mutation_fn = jax.vmap(mutation_fn)

	# shard the (embarrassingly parallel) evaluation across devices
	if nb_devices > 1 and N % nb_devices == 0 and nb_init_genotypes % nb_devices == 0:
		mesh = jax.make_mesh((nb_devices,), ("N",))
		sharding = NamedSharding(mesh, PartitionSpec(("N",)))
		batched_eval_fn = jax.jit(batched_eval_fn, in_shardings=(sharding, sharding), out_shardings=sharding)
		batched_mutation_fn = jax.jit(batched_mutation_fn, out_shardings=sharding)
	else:
		batched_eval_fn = jax.jit(batched_eval_fn)
		batched_mutation_fn = jax.jit(batched_mutation_fn)

	# ---
	@eqx.filter_jit
	def _train_step(state: TrainState, step) -> tuple[TrainState, dict | None]:
		new_key, key_sample, key_mut, key_task, key_add = jr.split(state.key, 5)

		sample = state.repertoire.select(key_sample, N)
		parents_infos = repertoire_cls.extract_parents_data(sample)

		genotypes = batched_mutation_fn(sample.genotypes, jr.split(key_mut, N), sample.genotype_infos)
		fitnesses, descriptors, _ = batched_eval_fn(genotypes, jr.split(key_task, N))

		repertoire, _ = state.repertoire.add(batch_of_genotypes=genotypes,
		                                     batch_of_fitnesses=fitnesses,
		                                     batch_of_descriptors=descriptors,
		                                     batch_of_task_extra_scores={},
		                                     batch_of_parents_infos=parents_infos,
		                                     key=key_add)

		state = state.replace(repertoire=repertoire, steps=state.steps + 1, key=new_key)
		metrics = qd_metrics(repertoire)

		if verbose:
			jax.debug.print("step {s} | " + " | ".join(f"{k}: {{{k}}}" for k in metrics), s=state.steps, **metrics)

		return state, (metrics if return_trace else None)
	# ---

	def _train(key: jax.Array):
		key_init, key_eval, key_rep, key_train = jr.split(key, 4)

		init_genotypes = jax.vmap(genotype_fctry)(jr.split(key_init, nb_init_genotypes))
		fitnesses, descriptors, _ = batched_eval_fn(init_genotypes, jr.split(key_eval, nb_init_genotypes))

		# task_extra_scores is left empty: the repertoire's `init` expects the [N, K, ...]
		# layout of a K-times-repeated evaluation, which a single eval per genotype has not got
		repertoire = repertoire_cls.init(batch_of_genotypes=init_genotypes,
										 batch_of_fitnesses=fitnesses,
										 batch_of_descriptors=descriptors,
										 batch_of_task_extra_scores=None,
										 centroids=centroids,
										 key=key_rep,
										 **repertoire_kwargs)

		init_state = TrainState(repertoire=repertoire, steps=0, key=key_train)
		return jax.lax.scan(_train_step, init_state, jnp.arange(train_steps))
	# ---

	return _train


def make_rand_genotype_fctry(model_kwargs: dict) -> tuple[Callable, RAND_CTRNN]:
	"""Return `(key -> RAND params, a template model)` for the given RAND config."""
	model = RAND_CTRNN(**model_kwargs, key=jr.key(0))
	fctry = lambda key: eqx.filter(RAND_CTRNN(**model_kwargs, key=key), eqx.is_array)
	return fctry, model


def develop_with_trace(genotype, model: RAND_CTRNN, key: jax.Array):
	"""Grow a RAND genotype, returning `(grown_network, dev_trajectory)`.

	`genotype` is the RAND parameter pytree stored in the archive (`repertoire.genotypes[i]`, what
	`make_rand_genotype_fctry`'s factory produces); `model` is the template from the same call,
	supplying the non-array structure. Development is stochastic, so to animate a rollout together
	with its growth you must reuse the *same* realization for both: grow once here, feed
	`grown_network` to the rollout via `neural_state=`, and pass `dev_trajectory` to
	`animate_multitaxis_rollout`. Re-growing from the genotype for the animation would draw a
	different network than the one that actually moved.

	The key is split exactly as `RAND_CTRNN.init` splits it, so `grown_network` is what
	`agent_interface.neural_init(genotype, key)` would return for this key — this just also hands
	back the intermediate developmental states (`do_migration`'s trace).
	"""
	model = eqx.combine(genotype, model)     # array leaves from the genotype, structure from model
	key_init, key_migr = jr.split(key)
	state = model.init_embryo(key_init)
	final, trace = model.do_migration(state, key_migr, with_trace=True)
	return model.make_network(final), trace["state"]
