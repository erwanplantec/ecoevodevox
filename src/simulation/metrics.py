from .core import SimulationState

import jax
import jax.numpy as jnp
import numpy as np


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
    arrays = {}       # masked per-agent arrays, kept for derived metrics
    reduced_data = {} # scalar reductions, always safe to log
    log_hist = {}     # values logged as-is (scalars + arrays wandb turns into histograms)
    for k, v in data.items():
        if k=="alive": continue
        if not v.shape:
            log_hist[k] = v
        elif v.shape[0]==mask.shape[0]:
            arr = v[mask]
            arrays[k] = arr
            reduced_data[f"{k} (avg)"] = np.mean(arr)
            reduced_data[f"{k} (max)"] = np.max(arr)
            reduced_data[f"{k} (min)"] = np.min(arr)
            reduced_data[f"{k} (var)"] = np.var(arr)
            # log the raw array (-> wandb histogram) only when it can be histogrammed; cast to
            # float64 so numpy can place 32 bin edges (float16 body/energy arrays cluster so
            # tightly that the edges collapse -> "Cannot create 32 finite-sized bins").
            if _is_histogrammable(arr):
                log_hist[k] = arr.astype(np.float64)
    # derived: per-agent reproduction rate (offsprings / age)
    if "offsprings" in arrays and "ages" in arrays:
        rr = (arrays["offsprings"] / np.clip(arrays["ages"], 1, None)).astype(np.float64)
        reduced_data["reproduction_rates (avg)"] = np.mean(rr)
        if _is_histogrammable(rr):
            log_hist["reproduction_rates"] = rr
    # derived: per-agent average speed (distance travelled / age)
    if "distance_travelled" in arrays and "ages" in arrays:
        speed = (arrays["distance_travelled"] / np.clip(arrays["ages"], 1, None)).astype(np.float64)
        reduced_data["speed (avg)"] = np.mean(speed)
        reduced_data["speed (max)"] = np.max(speed)
        if _is_histogrammable(speed):
            log_hist["speed"] = speed
    return {**log_hist, **reduced_data}


def _is_histogrammable(arr):
    """Whether wandb can build a histogram from `arr` without erroring. Needs >1 element and,
    in float64 (the dtype it is logged as), no non-finite values and 32 distinct bin edges —
    i.e. a spread wide enough that np.histogram(_, 32) succeeds."""
    if arr.size <= 1:
        return False
    arr = np.asarray(arr, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        return False
    try:
        np.histogram(arr, bins=32)
    except ValueError:
        return False
    return True