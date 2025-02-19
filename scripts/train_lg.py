from functools import partial
from src.devo.policy import CTRNNPolicyConfig
from src.devo.model_lg import Model_LG, mutate
from src.utils.viz import render_network
from src.evo.ga import GA

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import equinox as eqx
import evosax as ex
import realax as rx
import numpy as np
import matplotlib.pyplot as plt
from typing import NamedTuple
import wandb


def random_key():
    return jr.key(np.random.randint(0, 1_000_000))

def make_obs_to_input_fn(input_embeddings, decay_rate=5., max_radius=0.5):
    # ---
    def o2I(net, obs):
        dist = jnp.linalg.norm(net.x[:,None] - input_embeddings[None], axis=-1)
        influence = jnp.exp(-dist*decay_rate) * (dist < max_radius)
        I = jnp.sum(influence * obs[None], axis=1) * net.s * net.mask
        return I
    # ---
    return o2I

def make_activations_to_action_fn(action_embeddings, decay_rate=5., is_discrete=True, max_radius=1.):
    # ---
    def a2a(net):
        dist = jnp.linalg.norm(net.x[:,None] - action_embeddings[None], axis=-1)
        influence = jnp.exp(-dist*decay_rate) * (dist < max_radius)
        act = jnp.sum(influence * net.a[:,None] * net.m[:, None] * net.mask[:,None], axis=0)
        return act if not is_discrete else jnp.argmax(act)
    # ---
    return a2a


class Config(NamedTuple):
    env: str="CartPole-v1"
    gens: int=16
    pop: int=64
    eval_reps: int=1
    log: bool=True
    p_duplicate: float=0.05
    sigma: float=0.01
    N0: int=8
    max_types: int=8
    max_N: int=256
    synaptic_markers: int=8
    latent_dims: int=16


def train(cfg: Config):
    obs_dims, action_dims, is_discrete = rx.ENV_SPACES[cfg.env]

    input_angles = jnp.linspace(jnp.pi/3, 2*jnp.pi/3, obs_dims)
    input_embeddings = jnp.stack([jnp.cos(input_angles), jnp.sin(input_angles)], axis=-1)
    encode_fn = make_obs_to_input_fn(input_embeddings)

    output_angles = jnp.linspace(-jnp.pi/3, -2*jnp.pi/3, action_dims) if action_dims>1 else jnp.array([-jnp.pi/2])
    output_embeddings = jnp.stack([jnp.cos(output_angles), jnp.sin(output_angles)], axis=-1) 
    decode_fn = make_activations_to_action_fn(output_embeddings, is_discrete=is_discrete)

    policy_cfg = CTRNNPolicyConfig(encode_fn, decode_fn)
    decoder_kws = dict(width=64, depth=1)
    policy = Model_LG(cfg.latent_dims, cfg.N0, cfg.max_N, cfg.max_types, synaptic_markers=cfg.synaptic_markers, 
        policy_config=policy_cfg, key=random_key(), decoder_kwargs=decoder_kws)

    # ---
    # net = policy.initialize(random_key())
    # fig, ax = plt.subplots(1, 2, figsize=(10,5))
    # obs = jnn.one_hot(0, obs_dims)
    # I = policy.encode(net, obs)
    # render_network(net, node_colors=plt.cm.rainbow(I), ax=ax[0])
    # ax[0].scatter(*input_embeddings.T, marker="*", s=300)
    # ax[0].scatter(*output_embeddings.T, marker="^", s=300)
    # action, net = policy(obs, net, random_key())
    # amin, amax = net.a.min(), net.a.max()
    # render_network(net, node_colors=plt.cm.rainbow((net.a-amin) / (amax-amin)), ax=ax[1])
    # plt.show()
    # ---

    prms, sttcs = eqx.partition(policy, eqx.is_array)
    prms = eqx.tree_at(lambda tree: tree.O, prms, jnp.zeros_like(prms.O))
    fctry = lambda prms: eqx.combine(prms, sttcs)
    params_shaper = ex.ParameterReshaper(prms, verbose=False)

    def metrics_fn(state, data):
        log_data, ckpt_data, ep = rx.default_es_metrics(state, data)
        policy_states = data["eval_data"]["policy_states"] # P, T, ...
        archive = state.archive
        prms = params_shaper.reshape(archive)
        log_data["active types"] = prms.types.active.sum(-1)
        log_data["network sizes"] = policy_states.mask[:,-1].sum(-1)
        # eval
        x = state.archive[0]
        prms = params_shaper.reshape_single(x)
        fs, _ = jax.vmap(tsk, in_axes=(None,0))(prms, jr.split(jr.key(1), 16))
        log_data["evaluation: avg fitness"] = fs.mean()
        log_data["evaluation: max_fitness"] = fs.max()
        policy = fctry(prms)
        net = policy.initialize(jr.key(1))
        log_data["evaluation: network size"] = net.mask.sum()
        log_data["evaluation: active types"] = policy.types.active.sum()
        return log_data, ckpt_data, ep
        
    tsk = rx.GymnaxTask(cfg.env, fctry)

    mut_msk = jax.tree.map(lambda x: jnp.ones_like(x), prms)
    mut_msk = eqx.tree_at(lambda tree: tree.types.active, mut_msk, jnp.zeros_like(prms.types.active))
    mut_msk = params_shaper.flatten_single(mut_msk)

    lower_bound, upper_bound = policy.prms_lower_bound(), policy.prms_upper_bound()
    lower_bound = params_shaper.flatten_single(lower_bound)
    upper_bound = params_shaper.flatten_single(upper_bound)
    clip_fn = lambda prms: jnp.clip(prms, lower_bound, upper_bound)

    mutation_fn = lambda x, k, s: (
        mutate(x, k, cfg.p_duplicate, s.sigma, params_shaper, mut_msk, clip_fn), 
        None
    )

    ga = GA(mutation_fn, prms, cfg.pop, elite_ratio=0.5, sigma_init=cfg.sigma, sigma_decay=1., sigma_limit=0.01, p_duplicate=cfg.p_duplicate)

    logger = rx.Logger(cfg.log, metrics_fn)
    trainer = rx.EvosaxTrainer(cfg.gens, ga, tsk, params_like=prms, eval_reps=cfg.eval_reps, logger=logger)

    if cfg.log: wandb.init(project="evodevox", config=dict())
    s = jax.block_until_ready(trainer.init_and_train_(random_key()))
    x = s.archive[0]
    prms = trainer.params_shaper.reshape_single(x)
    policy = fctry(prms)
    net = policy.initialize(random_key())
    fig = plt.figure()
    ax = fig.add_subplot()
    render_network(net, ax=ax)
    if cfg.log:
        wandb.log({"best network": wandb.Image(fig)})
        wandb.finish()

if __name__ == '__main__':
    cfg = Config(pop=8, gens=8)
    with jax.debug_nans(): train(cfg)