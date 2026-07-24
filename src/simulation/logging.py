from .core import SimulationState

import os
from typing import Callable
import wandb
import datetime
try:
    import _pickle as pickle
except:
    import pickle
import jax, jax.numpy as jnp, jax.random as jr
from jax.experimental import io_callback
import random
import string
import shutil
import os

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
                 log_freq: int=1,  # Log metrics every `log_freq` steps (1 = every step)
                 ckpt_freq: int|None=None,  # Frequency of checkpoint saves (in steps)
                 sampling_freq: int|None=None, # Frequency of sampling
                 sampling_size: int=16, # size of samples
                 plot_networks: int=0, # #living agents whose grown network is rendered to wandb at the end
                 metrics_fn: Callable=metrics_fn, # Function to compute metrics (executed on device side)
                 host_log_transform: Callable=host_log_transform, # Function to transform metrics for logging on host side,
                 wandb_project: str="eedx"): 
        if name:
            name = name
        else:
            name = get_date_string()
        os.makedirs("data", exist_ok=True)
        if name in os.listdir("data"):
            print(f"name {name} is already used (found in data folder).")
            name = get_date_string()
        self.name = name
        print(f"Instantiating simulator. name: {self.name}")
            
        # ----
        self.wandb_log = wandb_log
        self.wandb_project = wandb_project
        self.log_freq = max(1, int(log_freq))
        self.plot_networks = int(plot_networks or 0)
        self.metrics_fn = metrics_fn
        # ---
        self.ckpt_freq = ckpt_freq
        if ckpt_freq is not None and ckpt_freq>0:
            ckpt_dir = f"data/{self.name}/ckpts"
            os.makedirs(ckpt_dir, exist_ok=True)
        else:
            ckpt_dir = None
        self.ckpt_dir = ckpt_dir
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
            if not wandb_log: return False
            transformed_data = host_log_transform(data)
            try: 
                wandb.log(transformed_data)
            except: 
                raise ValueError("logging")
            return False
        self._log_clbk = _log_clbk

        def _ckpt_clbk(sim_state):
            if ckpt_freq is None: return jnp.zeros((), dtype=bool)
            time = sim_state.env_state.time
            filename = f"{ckpt_dir}/{int(time)}.pickle"
            with open(filename, "wb") as file:
                pickle.dump(sim_state, file)
            return False
        self._ckpt_clbk = _ckpt_clbk

        def _sample_clbk(sample, time):
            if sampling_freq is None: return False
            filename = f"{sampling_dir}/{int(time)}.pickle"
            with open(filename, "wb") as file:
                pickle.dump(sample, file)
            return False

        self._sample_clbk = _sample_clbk
    # ---

    def initialize(self, cfg: dict):

        if self.wandb_log:
            wandb.init(project=self.wandb_project, name=self.name, config=cfg)

    # ---

    def log(self, sim_state: SimulationState, step_data, key):
        
        # --- 1. data logging (every `log_freq` steps) ---
        if self.wandb_log:

            # gate the host callback like the ckpt/sample paths below: on non-logging steps the
            # io_callback is skipped, so there is no device->host sync stalling the rollout. This is
            # what makes throughput scale back up when logging is sparse.
            def _do_log(operands):
                sim_state, step_data = operands
                data = self.metrics_fn(sim_state, step_data)
                return io_callback(self._log_clbk, jnp.zeros((), dtype=bool), data)

            jax.lax.cond(
                jnp.mod(sim_state.time, self.log_freq) == 0,
                _do_log,
                lambda operands: jnp.zeros((), dtype=bool),
                (sim_state, step_data),
            )
        
        # --- 2. do ckpt ---
        if self.ckpt_dir is not None:

            assert isinstance(self.ckpt_freq, int)

            _ = jax.lax.cond(
                jnp.mod(sim_state.time, self.ckpt_freq)==0,
                lambda s: io_callback(self._ckpt_clbk, jax.ShapeDtypeStruct((), bool), s),
                lambda *a, **k: jnp.zeros((), dtype=bool),
                sim_state
            )

        # --- 3. save sample of agents ---
        if self.sampling_dir is not None:

            assert isinstance(self.sampling_freq, int)

            def _sample_and_clbk(agents, time, key):
                p = agents.alive / agents.alive.sum()
                sample_ids = jr.choice(key, agents.alive.shape[0], shape=(self.sampling_size,), p=p)
                sample = jax.tree.map(lambda x: x[sample_ids], agents)
                return io_callback(self._sample_clbk, jax.ShapeDtypeStruct((),bool), sample, time)

            _ = jax.lax.cond(
                jnp.mod(sim_state.time, self.sampling_freq)==0,
                _sample_and_clbk,
                lambda *a, **k: jnp.zeros((), dtype=bool),
                sim_state.agents_states, sim_state.time, key
            )
    # ---

    def log_networks(self, sim_state: SimulationState, key, n: int|None=None, col_wrap: int=5):
        """Render a sample of *living* agents' grown networks into a single grid figure and log it
        to wandb as one image.

        Only works for spatially-embedded / grown encodings, whose neural_state carries `x`, `W`
        and `mask` (RAND, NeuronNCA). No-op otherwise, or when wandb is off / nobody is alive.
        Meant to be called once at the end of a run.
        """
        import numpy as np
        n = self.plot_networks if n is None else int(n)
        if not self.wandb_log or n <= 0:
            return
        agents = sim_state.agents_states
        ns = agents.neural_state
        if not all(hasattr(ns, a) and getattr(ns, a) is not None for a in ("x", "W", "mask")):
            return
        try:
            import matplotlib
            matplotlib.use("Agg")               # headless: no display needed
            import matplotlib.pyplot as plt
            from ..utils.viz import render_network
        except Exception as e:
            print(f"log_networks: plotting unavailable ({e})")
            return

        alive = np.asarray(agents.alive)
        idx = np.where(alive)[0]
        if idx.size == 0:
            return
        rng = np.random.default_rng(int(jr.randint(key, (), 0, 2**31 - 1)))
        sel = rng.choice(idx, size=min(n, idx.size), replace=False)
        gens = np.asarray(agents.generation)
        offs = np.asarray(agents.n_offsprings)

        ncols = min(col_wrap, len(sel))
        nrows = -(-len(sel) // ncols)           # ceil div
        fig, axs = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), squeeze=False)
        axs = axs.ravel()
        for k, ax in enumerate(axs):
            if k >= len(sel):
                ax.set_visible(False)           # hide unused cells in the last row
                continue
            i = int(sel[k])
            net_i = jax.tree.map(lambda a: a[i], ns)
            render_network(net_i, ax=ax)
            ax.set_aspect("equal"); ax.axis("off")
            ax.set_title(f"agent {i} · gen {int(gens[i])} · offs {int(offs[i])}", fontsize=8)
        fig.tight_layout()
        wandb.log({"evolved_networks": wandb.Image(fig)})
        plt.close(fig)
        print(f"log_networks: logged {len(sel)} networks to wandb")

    # ---

    def finish(self):
        if self.wandb_log:
            wandb.finish()
        if self.ckpt_dir is not None or self.sampling_dir is not None:
            data_dir = f"data/{self.name}"
            shutil.make_archive(data_dir, "zip", "data", self.name)
            shutil.rmtree(data_dir, ignore_errors=True)

        

