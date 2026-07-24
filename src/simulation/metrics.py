from .core import SimulationState

import jax
import jax.numpy as jnp
import numpy as np


def food_patchiness(food, quadrat=8, min_cells=20):
    """Spatial patchiness of a binary food map (host-side, numpy), independent of density.

    `food` is a 2D boolean/0-1 array (one food channel, or the union of channels). The world is
    toroidal and both metrics respect that wrap. Returns ``(correlation_length, vmr_norm)``:

    - **correlation_length** — the 1/e radius (in cells) of the FFT spatial autocorrelation of the
      connected correlation ``g(d) = (C(d) - p^2)/(p - p^2)``. This is the characteristic *patch
      size*: ~0 for scattered/random food, growing to the blob radius for clustered food. Being
      built from `g`, it is density-independent (verified: uniform-random food gives the same ~0.6
      at any density).
    - **vmr_norm** — the variance-to-mean ratio of food counts over ``quadrat x quadrat`` blocks,
      divided by ``(1 - p)``. The division corrects the binary-occupancy baseline (a random binary
      field has raw VMR = 1-p, not 1), so vmr_norm is ~1 for random, >1 for clumped, <1 for regular
      / over-dispersed. It is scale-dependent — `quadrat` is the scale it probes.

    Returns ``(nan, nan)`` when the map is empty, full, or has fewer than `min_cells` food cells
    (patchiness is ill-defined there).
    """
    f = np.asarray(food, dtype=np.float64)
    H, W = f.shape
    p = f.mean()
    if f.sum() < min_cells or p <= 0.0 or p >= 1.0:
        return float("nan"), float("nan")

    # --- correlation length via FFT autocorrelation (circular == toroidal) ---
    F = np.fft.fft2(f)
    C = np.fft.ifft2(np.abs(F) ** 2).real / (H * W)      # C[0,0] = p
    g = (C - p * p) / (p - p * p)                         # g[0,0] = 1, g[far] -> 0
    ii = np.minimum(np.arange(H), H - np.arange(H))
    jj = np.minimum(np.arange(W), W - np.arange(W))
    dist = np.hypot(ii[:, None], jj[None, :])            # toroidal distance from zero lag
    rmax = int(min(H, W) // 2)
    rb = np.clip(np.round(dist).astype(int), 0, rmax)
    # radial average of g by integer distance, vectorised (a Python loop over rmax boolean masks
    # of the full grid is O(rmax*H*W) and dominates on large worlds — 0.3s on 1024^2)
    sums = np.bincount(rb.ravel(), weights=g.ravel(), minlength=rmax + 1)[: rmax + 1]
    cnts = np.bincount(rb.ravel(), minlength=rmax + 1)[: rmax + 1]
    gr = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)
    thr = 1.0 / np.e
    corr_length = float("nan")
    for r in range(1, rmax + 1):
        if gr[r] < thr:
            g0, g1 = gr[r - 1], gr[r]
            corr_length = (r - 1) + (g0 - thr) / (g0 - g1) if g0 != g1 else float(r)
            break

    # --- normalized variance-to-mean ratio over quadrats ---
    q = int(quadrat)
    fc = f[: H // q * q, : W // q * q]
    counts = fc.reshape(fc.shape[0] // q, q, fc.shape[1] // q, q).sum(axis=(1, 3))
    m = counts.mean()
    vmr_norm = float(counts.var() / m / (1.0 - p)) if m > 0 and p < 1.0 else float("nan")

    return float(corr_length), vmr_norm


def metrics_fn(sim_state: SimulationState, step_data: dict)->dict:
    """computes metrics from world state and step_data to log"""
    agents = sim_state.agents_states
    env = sim_state.env_state
    food_levels = {
        f"food_{i}": env.food[i].sum() for i in range(env.food.shape[0])
    }
    data = {
        "alive": agents.alive,
        "population": agents.alive.sum(),
        "energy_levels": agents.energy,
        "offsprings": agents.n_offsprings,
        "distance_travelled": agents.distance_travelled,
        "total_abs_turn": agents.total_abs_turn,
        **step_data,
        "ages": agents.age,
        **food_levels,
        "total_food": env.food.sum(),
        "body_sizes": agents.body.size
    }
    # per-agent neuron count, when the neural model exposes a developmental mask
    # (grown encodings: RAND, NeuronNCA). Direct encodings (ctrnn/rnn) have no mask.
    if hasattr(agents.neural_state, "mask") and agents.neural_state.mask is not None:
        data["nb_neurons"] = agents.neural_state.mask.sum(-1)
    return data

def host_log_transform(data: dict)->dict:
    data = jax.tree.map(np.asarray, data)
    mask = data["alive"]
    if not np.any(mask):
        return {"population": 0}
    arrays = {}       # masked per-agent arrays, kept only to derive the scalar metrics below
    reduced_data = {} # scalar reductions, always safe to log
    scalars = {}      # already-scalar metrics (population, total_food, food_i, ...)
    for k, v in data.items():
        if k=="alive": continue
        if not v.shape:
            scalars[k] = v
        elif v.shape[0]==mask.shape[0]:
            arr = v[mask]
            arrays[k] = arr
            reduced_data[f"{k} (avg)"] = np.mean(arr)
            reduced_data[f"{k} (max)"] = np.max(arr)
            reduced_data[f"{k} (min)"] = np.min(arr)
            reduced_data[f"{k} (var)"] = np.var(arr)
            # per-agent arrays are logged only as these scalar reductions — the raw arrays (wandb
            # histograms) were the heaviest part of the host payload, so they are dropped.
    # derived: per-agent reproduction rate (offsprings / age)
    if "offsprings" in arrays and "ages" in arrays:
        rr = arrays["offsprings"] / np.clip(arrays["ages"], 1, None)
        reduced_data["reproduction_rates (avg)"] = np.mean(rr)
    # derived: per-agent average speed (distance travelled / age)
    if "distance_travelled" in arrays and "ages" in arrays:
        speed = arrays["distance_travelled"] / np.clip(arrays["ages"], 1, None)
        reduced_data["speed (avg)"] = np.mean(speed)
        reduced_data["speed (max)"] = np.max(speed)
    # derived: per-agent average angular speed (summed |heading change| / age)
    if "total_abs_turn" in arrays and "ages" in arrays:
        ang = arrays["total_abs_turn"] / np.clip(arrays["ages"], 1, None)
        reduced_data["angular_speed (avg)"] = np.mean(ang)
        reduced_data["angular_speed (max)"] = np.max(ang)
    return {**scalars, **reduced_data}