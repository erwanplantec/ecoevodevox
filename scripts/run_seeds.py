"""Run N seeds of a config sequentially, each as its own wandb run.

Each seed gets a fresh Logger (so its samples/ckpts land in a distinct data/<name> dir and it
opens a separate wandb run) and a fresh Simulator (the logger's callbacks are captured when the
step is compiled, so the simulator must be rebuilt to pick up the new logger). The world and agent
interface are seed-independent, so they are built once and shared.

Pin to a single GPU from the caller, e.g.:
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_seeds.py configs/rand_baseline.yml \
        --seeds $(seq 0 19) --steps 200000 --name-prefix rand_baseline
"""
import argparse
import time

import jax
import jax.random as jr

from src.simulation.utils import load_config_file, make_world, make_agents_interface
from src.simulation.core import SimulationConfig
from src.simulation.simulation import Simulator
from src.simulation.logging import Logger
from src.simulation.metrics import metrics_fn, host_log_transform


def run(config: str, seeds: list[int], steps: int, name_prefix: str):
    cfg = load_config_file(config)
    world, _ = make_world(cfg)
    agent_interface, mutation_fn = make_agents_interface(cfg)
    sim_cfg = SimulationConfig(**cfg["simulation"])
    log_cfg = cfg["logging"]

    print(f"config={config} seeds={seeds} steps={steps} devices={jax.devices()}", flush=True)
    t_all = time.time()
    for i, seed in enumerate(seeds):
        name = f"{name_prefix}_s{seed}"
        run_cfg = {**cfg, "seed": seed}          # so the seed is recorded in the wandb config
        logger = Logger(wandb_log=log_cfg.get("wandb_log", False),
                        name=name,
                        log_freq=log_cfg.get("log_freq", 1),
                        ckpt_freq=log_cfg.get("ckpt_freq", None),
                        sampling_freq=log_cfg.get("sampling_freq", None),
                        sampling_size=log_cfg.get("sampling_size", 16),
                        plot_networks=log_cfg.get("plot_networks", 0),
                        metrics_fn=metrics_fn,
                        host_log_transform=host_log_transform,
                        wandb_project=log_cfg.get("wandb_project", "eedx"))
        logger.initialize(run_cfg)
        sim = Simulator(cfg=sim_cfg, world=world, agent_interface=agent_interface,
                        mutation_fn=mutation_fn, logger=logger)

        key_init, key_rollout = jr.split(jr.key(seed))
        t = time.time()
        state = sim.initialize(key=key_init)
        state, _ = sim.rollout(state, steps, key=key_rollout)
        jax.block_until_ready(state.time)
        logger.log_networks(state, jr.fold_in(key_rollout, 999))   # grid of evolved networks
        sim.finish()
        print(f"[{i + 1}/{len(seeds)}] seed={seed} name={name} "
              f"done in {(time.time() - t) / 60:.1f} min, "
              f"final_time={int(state.time)}/{steps}", flush=True)
    print(f"ALL DONE ({len(seeds)} seeds) in {(time.time() - t_all) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("config", type=str)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--name-prefix", type=str, default="run")
    a = p.parse_args()
    run(a.config, a.seeds, a.steps, a.name_prefix)
