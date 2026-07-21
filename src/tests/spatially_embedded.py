"""Tests for the spatially-embedded sensory interface.

Run it directly:

    uv run python src/tests/spatially_embedded.py

The interface (src/devo/sensory/spatially_embedded.py) works like this:

  * init() maps each neuron position x in [-1, 1]^2 to a grid cell
        cell = floor((x + 1) / 2.0001 * body_resolution)
    and flags a neuron as "on border" if any |coord| > border_threshold.
  * encode() stacks the environment field with the internal signals (tiled over space)
    into `inp` of shape [C_env + n_internal, R, R], then for each on-border neuron i reads
    the channel vector at its cell and dots it with the neuron's sensitivity s_i:
        I_i = sum_c inp[c, cell_i] * s_i[c]          (0 if the neuron is not on border)

So the interface requires s_i to have one entry per input channel, i.e.
    sensory_genes == C_env + n_internal == (n_chemicals + walls) + n_internal.

The checks below place neurons at known cells and drive them with hot cells / uniform
internal signals whose outputs are predictable by hand, then cross-check a random case
against an explicit reference implementation.
"""

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
	sys.path.insert(0, _REPO_ROOT)

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from src.devo.core import Observation
from src.devo.sensory.spatially_embedded import SpatiallyEmbeddedSensoryInterface


class FakeNeuralState(struct.PyTreeNode):
	x: jax.Array       # [N, 2] positions in [-1, 1]^2
	v: jax.Array       # [N]    activations (asserted present, unused by encode)
	s: jax.Array       # [N, C] per-channel sensitivity
	mask: jax.Array    # [N]    alive/present


# ---------------------------------------------------------------------------
# reference implementation, deliberately a plain numpy transcription of the code
# ---------------------------------------------------------------------------
def ref_encode(x, s, mask, env, internal, R, border_threshold, threshold):
	x = np.asarray(x, float); s = np.asarray(s, float); mask = np.asarray(mask, bool)
	env = np.asarray(env, float); internal = np.asarray(internal, float)
	N = x.shape[0]

	se = s * mask[:, None]
	se = np.where(se > threshold, se, 0.0)                       # sensory_expression

	coords = np.floor((x + 1) / 2.0001 * R).astype(int)          # [N, 2]
	on_border = np.any(np.abs(x) > border_threshold, axis=-1)    # [N]

	inp = np.concatenate([env, np.tile(internal[:, None, None], (1, R, R))], axis=0)  # [Ctot, R, R]

	I = np.zeros(N)
	for i in range(N):
		ci, cj = coords[i, 0], coords[i, 1]
		I[i] = np.sum(inp[:, ci, cj] * se[i]) if on_border[i] else 0.0
	return I, coords, on_border


def _interface(R, threshold=0.0, border_threshold=0.0, energy_cost=0.0):
	return SpatiallyEmbeddedSensoryInterface(
		body_resolution=R, sensor_expression_threshold=threshold,
		border_threshold=border_threshold, sensor_energy_cost=energy_cost,
	)


def _encode(itf, state, env, internal):
	layout = itf.init(state, jax.random.key(0))
	obs = Observation(env=jnp.asarray(env), internal=jnp.asarray(internal))
	I, cost, _, _ = itf.encode(obs, state, layout)
	return np.asarray(I), np.asarray(layout.indices), np.asarray(layout.on_border), float(cost)


def run_checks() -> bool:
	R = 4
	cases = []

	# neurons at cell centres of a 4x4 grid, plus one exactly at the origin
	#            cell(0,0)      cell(3,0)     cell(0,3)     cell(2,2)     origin -> cell(1,1)
	pos = jnp.array([[-.75, -.75], [.75, -.75], [-.75, .75], [.25, .25], [0., 0.]])
	N = pos.shape[0]

	# --- 1. coordinate mapping ---
	st = FakeNeuralState(x=pos, v=jnp.zeros(N), s=jnp.ones((N, 1)), mask=jnp.ones(N, bool))
	env0 = jnp.zeros((1, R, R)); internal0 = jnp.zeros(1)  # s width 1 = 1 env channel + 0 internal
	# use a 1-channel env with NO internal here so s width (1) matches
	_, idx, on_border, _ = _encode(_interface(R), st, env0, internal0)
	expected_cells = np.array([[0, 0], [3, 0], [0, 3], [2, 2], [1, 1]])
	cases.append(("neuron positions map to expected cells", np.array_equal(idx, expected_cells)))

	# --- 2. on_border (default border_threshold=0.0 excludes only the exact origin) ---
	cases.append(("off-centre neurons are on border", bool(np.all(on_border[:4]))))
	cases.append(("origin neuron is NOT on border", not bool(on_border[4])))

	# --- 3. a single hot env cell drives only the neuron that maps to it ---
	env = np.zeros((1, R, R)); env[0, 3, 0] = 5.0  # cell (3,0) == neuron 1
	I, *_ = _encode(_interface(R), st, env, np.zeros(1))
	cases.append(("hot cell (3,0) drives only neuron 1 = s*value",
	              np.allclose(I, [0, 5, 0, 0, 0])))

	# --- 4. sensitivity weighting: same hot cell, neuron-1 sensitivity halved ---
	s_half = jnp.ones((N, 1)).at[1, 0].set(0.5)
	st_half = st.replace(s=s_half)
	I, *_ = _encode(_interface(R), st_half, env, np.zeros(1))
	cases.append(("output scales with sensitivity (0.5 * 5 = 2.5)", np.isclose(I[1], 2.5)))

	# --- 5. sub-threshold sensitivity is zeroed ---
	s_low = jnp.full((N, 1), 0.05)
	st_low = st.replace(s=s_low)
	I, *_ = _encode(_interface(R, threshold=0.1), st_low, env, np.zeros(1))
	cases.append(("sensitivity below threshold -> no response", np.allclose(I, 0.0)))

	# --- 6. dead (masked) neuron does not respond ---
	st_dead = st.replace(mask=jnp.array([True, False, True, True, True]))
	I, *_ = _encode(_interface(R), st_dead, env, np.zeros(1))
	cases.append(("masked neuron 1 silent despite hot cell", np.isclose(I[1], 0.0)))

	# --- 7. internal signals contribute uniformly across space ---
	# s width 3 = 1 env channel + 2 internal; env all zero, internal = [2, 3]
	s3 = jnp.ones((N, 3))
	st3 = FakeNeuralState(x=pos, v=jnp.zeros(N), s=s3, mask=jnp.ones(N, bool))
	I, *_ = _encode(_interface(R), st3, np.zeros((1, R, R)), np.array([2.0, 3.0]))
	# every on-border neuron sees the same internal sum 2+3=5; the origin neuron stays 0
	cases.append(("internal signals add uniformly to border neurons",
	              np.allclose(I, [5, 5, 5, 5, 0])))

	# --- 8. energy cost = sensor_energy_cost * sum(expressed sensitivity) ---
	_, _, _, cost = _encode(_interface(R, energy_cost=0.1), st, env, np.zeros(1))
	cases.append(("energy cost = cost * sum(s)", np.isclose(cost, 0.1 * N, atol=1e-2)))

	# --- 9. full linearity vs explicit reference, random inputs ---
	rng = np.random.default_rng(0)
	C_env, n_int = 2, 4
	xr = rng.uniform(-1, 1, (12, 2))
	sr = rng.uniform(-1, 1, (12, C_env + n_int))
	maskr = rng.random(12) > 0.3
	envr = rng.uniform(-1, 1, (C_env, R, R))
	intr = rng.uniform(-1, 1, n_int)
	str_ = FakeNeuralState(x=jnp.asarray(xr), v=jnp.zeros(12),
	                       s=jnp.asarray(sr), mask=jnp.asarray(maskr))
	# threshold below the data range so nothing is clipped -> tests pure linearity
	itf = _interface(R, threshold=-1e9)
	I, idx, ob, _ = _encode(itf, str_, envr, intr)
	I_ref, idx_ref, ob_ref = ref_encode(xr, sr, maskr, envr, intr, R, 0.0, -1e9)
	cases.append(("random case: indices match reference", np.array_equal(idx, idx_ref)))
	cases.append(("random case: on_border matches reference", np.array_equal(ob, ob_ref)))
	cases.append(("random case: encoded input matches reference", np.allclose(I, I_ref, atol=1e-4)))

	# --- 10. border_threshold gates who senses (near-edge only) ---
	# with border_threshold=0.9 only neurons with |coord|>0.9 are on border
	near_edge = jnp.array([[.95, 0.], [0.1, 0.1], [-.99, .2], [0., 0.]])
	stb = FakeNeuralState(x=near_edge, v=jnp.zeros(4), s=jnp.ones((4, 1)), mask=jnp.ones(4, bool))
	_, _, ob, _ = _encode(_interface(R, border_threshold=0.9), stb, np.zeros((1, R, R)), np.zeros(1))
	cases.append(("border_threshold=0.9 keeps only near-edge neurons on border",
	              np.array_equal(ob, [True, False, True, False])))

	# --- 11. determinism ---
	a = _encode(_interface(R), st, env, np.zeros(1))[0]
	b = _encode(_interface(R), st, env, np.zeros(1))[0]
	cases.append(("encode is deterministic", np.array_equal(a, b)))

	print("\n=== spatially-embedded sensory interface checks ===")
	ok = True
	for name, passed in cases:
		print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
		ok = ok and bool(passed)
	print(f"=== {'ALL PASSED' if ok else 'FAILURES PRESENT'} ===\n")
	return ok


if __name__ == "__main__":
	sys.exit(0 if run_checks() else 1)
