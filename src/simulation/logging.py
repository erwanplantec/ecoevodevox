import os
from typing import Callable
import wandb
import datetime
try:
    import _pickle as pickle
except:
    import pickle
import jax, jax.numpy as jnp
from jax.experimental import io_callback
import random
import string

from .metrics import host_log_transform, metrics_fn

def generate_random_name(length=8):
    return "".join([random.choice(string.ascii_lowercase+string.digits) for _ in range(length)])

def get_date_string():
    date = datetime.datetime.now()
    return date.strftime("%d_%m_%Y_%H:%M:%S")


class Logger:
    # ---
    def __init__(self, 
                 wandb_log: bool=False,  # Whether to log metrics during simulation
                 name: str|None=None,
                 ckpt_freq: int|None=None,  # Frequency of checkpoint saves (in steps)
                 sampling_freq: int|None=None, # Frequency of sampling
                 sampling_size: int=16, # size of samples
                 metrics_fn: Callable=metrics_fn, # Function to compute metrics (executed on device side)
                 host_log_transform: Callable=host_log_transform, # Function to transform metrics for logging on host side,
                 wandb_project: str="eedx"): 
        if name:
            name = name
        else:
            name = get_date_string()
        if name in os.listdir("data"):
            print(f"name {name} is already used (found in data folder).")
            name = get_date_string()
        self.name = name
        print(f"Instantiating simulator. name: {self.name}")
            
        # ----
        self.wandb_log = wandb_log
        self.wandb_project = wandb_project
        # ---
        self.ckpt_freq = ckpt_freq
        if ckpt_freq is not None and ckpt_freq>0:
            ckpt_dir = f"data/{self.name}/ckpts"
            os.makedirs(ckpt_dir, exist_ok=True)
        else:
            ckpt_dir = None
        self.ckpt_dit = ckpt_dir
        # ---
        self.sampling_freq = sampling_freq
        self.sampling_size = sampling_size
        if sampling_freq is not None and sampling_freq>0:
            sampling_dir = f"data/{name}/samples"
            os.makedirs(sampling_dir, exist_ok=True)
        else:
            sampling_dir = None
        self.sampling_dir = sampling_dir
        # ---

        def _log_clbk(data):
            if not wandb_log: return jnp.zeros((), dtype=bool)
            data = host_log_transform(data)
            wandb.log(data)
            return jnp.zeros((), dtype=bool)
        self._log_clbk = _log_clbk

        def _ckpt_clbk(sim_state):
            if ckpt_freq is None: return jnp.zeros((), dtype=bool)
            time = sim_state.env_state.time
            filename = f"{ckpt_dir}/{int(time)}.pickle"
            with open(filename, "wb") as file:
                pickle.dump(sim_state, file)
            return jnp.zeros((),dtype=bool)
        self._ckpt_clbk = _ckpt_clbk

        def _sample_clbk(sample, time):
            if sampling_freq is None: return jnp.zeros((), dtype=bool)
            filename = f"{sampling_dir}/{int(time)}.pickle"
            with open(filename, "wb") as file:
                pickle.dump(sample, file)
            return jnp.zeros((),dtype=bool)
        self._sample_clbk = _sample_clbk
    # ---

    def initialize(self, cfg: dict):

        if self.wandb_log:
            wandb.init(project=self.wandb_project)

    # ---

    def log(self, sim_state, step_data):
        if log:
            data = metrics_fn(sim_state, step_data)
            io_callback(self._log_clbk, jax.ShapeDtypeStruct((),bool), data)
        if ckpt_dir:
            assert isinstance(self.ckpt_freq, int)
            _ = jax.lax.cond(
                jnp.mod(state.time, ckpt_freq)==0,
                lambda s: io_callback(self._ckpt_clbk, jax.ShapeDtypeStruct((),bool), s),
                lambda *a, **k: jnp.zeros((), dtype=bool),
                {"env_state": state, "key": key}
            )
        if sampling_dir:
            assert isinstance(sampling_freq, int)
            def _sample_and_clbk(agents, time, key):
                p = agents.alive / agents.alive.sum()
                sample_ids = jr.choice(key, agents.alive.shape[0], shape=(sampling_size,), p=p)
                sample = jax.tree.map(lambda x: x[sample_ids], agents)
                return io_callback(_sample_clbk, jax.ShapeDtypeStruct((),bool), sample, time)
            _ = jax.lax.cond(
                jnp.mod(state.time, sampling_freq)==0,
                _sample_and_clbk,
                lambda *a, **k: jnp.zeros((), dtype=bool),
                state.agents_states, state.time, key
            )
        

