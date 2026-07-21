"""Saving and loading simulation checkpoints.

A checkpoint is a pickled dict holding the full `SimulationState` plus free-form metadata
(config, step count, ...). Arrays are pulled off-device before pickling and put back on load,
so a checkpoint is portable: it can be reloaded on a different number of devices or a
different sharding than it was written with.

    from src.simulation.checkpoint import save_state, load_state
    save_state("data/run.ckpt", sim_state, meta={"cfg": cfg, "step": 1000})
    sim_state, meta = load_state("data/run.ckpt")

The state only makes sense for a simulator built with the same shapes (max_agents, world
size, neural model), so store the config in `meta` and check it before reloading.
"""

import os
try:
    import _pickle as pickle
except ImportError:
    import pickle

import jax
import jax.numpy as jnp

from .core import SimulationState

_MAGIC = "eedx-checkpoint-v1"


def save_state(path: str, sim_state: SimulationState, meta: dict | None = None):
    """Pickle `sim_state` (+ metadata) to `path`, creating parent directories as needed."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    payload = {
        "magic": _MAGIC,
        # device_get -> numpy, so the checkpoint does not carry device/sharding assumptions
        "sim_state": jax.device_get(sim_state),
        "meta": meta or {},
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    return path


def load_state(path: str) -> tuple[SimulationState, dict]:
    """Load a checkpoint written by `save_state`.

    Also accepts a bare pickled `SimulationState` (what `Logger`'s periodic checkpoints
    write), in which case the returned metadata is empty.
    """
    with open(path, "rb") as f:
        payload = pickle.load(f)

    if isinstance(payload, dict) and payload.get("magic") == _MAGIC:
        sim_state, meta = payload["sim_state"], payload.get("meta", {})
    else:
        sim_state, meta = payload, {}          # bare state (logger checkpoint)

    # move arrays back onto the default device(s)
    sim_state = jax.tree.map(jnp.asarray, sim_state)
    return sim_state, meta
