from functools import partial

from jax.flatten_util import ravel_pytree
from src.devo.model_e import *
from src.utils.viz import render_network
from src.evo.ga import GA

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import equinox as eqx
import realax as rx
import numpy as np
import matplotlib.pyplot as plt
import wandb


def random_key():
    return jr.key(np.random.randint(0, 1_000_000))

def make_obs_to_input_fn(input_embeddings, decay_rate=5., max_radius=0.5):
    # ---
    def o2I(net, obs):
        dist = jnp.linalg.norm(net.x[:,None] - input_embeddings[None], axis=-1) #N,O
        influence = jnp.exp(-dist*decay_rate) * (dist < max_radius) #N,O
        I = jnp.sum(influence * obs[None], axis=1) * net.s[:,0] * net.mask
        return I
    # ---
    return o2I

def make_activations_to_action_fn(action_embeddings, decay_rate=5., is_discrete=True, max_radius=1.):
    # ---
    def a2a(net: CTRNN):
        # ---
        assert net.m is not None
        assert net.mask is not None
        # ---
        dist = jnp.linalg.norm(net.x[:,None] - action_embeddings[None], axis=-1)
        influence = jnp.exp(-dist*decay_rate) * (dist < max_radius)
        act = jnp.sum(influence * net.v[:,None] * net.m[:, None] * net.mask[:,None], axis=0)
        return act if not is_discrete else jnp.argmax(act)
    # ---
    return a2a


class Config(NamedTuple):
    env: str="CartPole-v1"
    gens: int=128
    pop: int=256
    eval_reps: int=1
    log: bool=True
    p_duplicate: float=0.1
    sigma: float=0.1
    N0: int=8


def train(cfg: Config):

    obs_dims, action_dims, is_discrete = rx.ENV_SPACES[cfg.env]

    input_embeddings = jnp.ones((obs_dims, 2)).at[:,0].set(jnp.linspace(-1, 1, obs_dims))
    action_embeddings = jnp.full((action_dims, 2), -1.).at[:,0].set(jnp.linspace(-1,1,action_dims))

    encode_fn = make_obs_to_input_fn(input_embeddings)
    decode_fn = make_activations_to_action_fn(action_embeddings)

    n_types = 8
    policy_cfg = CTRNNPolicyConfig(encode_fn, decode_fn)
    policy = Model_E(n_types, 8, dt=0.1, dvpt_time=10., policy_cfg=policy_cfg, key=random_key())
    policy = make_two_types(policy, 8, 2)

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
    init_prms = prms
    fctry = lambda prms: eqx.combine(prms, sttcs)
    params_shaper = ex.ParameterReshaper(prms, verbose=False)

    def metrics_fn(state, data):
        log_data, ckpt_data, ep = rx.default_es_metrics(state, data)
        policy_states = data["eval_data"]["policy_states"] # P, T, ...
        archive = state.archive
        prms = params_shaper.reshape(archive)
        log_data["active types"] = prms.types.active.sum(-1) #type:ignore
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


    mutation_mask = jax.tree.map(lambda x: jnp.ones_like(x), init_prms)
    mutation_mask = eqx.tree_at(lambda t: t.types.active, mutation_mask, jnp.zeros_like(init_prms.types.active))
    mutation_mask = eqx.tree_at(lambda t: t.types.id_, mutation_mask, jnp.zeros_like(init_prms.types.id_))
    mutation_mask = params_shaper.flatten_single(mutation_mask)
    clip_min = jax.tree.map(lambda x: jnp.full_like(x, -jnp.inf), init_prms)
    clip_min = eqx.tree_at(lambda tree: tree.types.gamma, clip_min, jnp.full_like(init_prms.types.gamma, 1e-8))
    clip_min = eqx.tree_at(lambda tree: tree.types.pi, clip_min, jnp.zeros_like(init_prms.types.pi))
    clip_min = params_shaper.flatten_single(clip_min)
    clip_max = jax.tree.map(lambda x: jnp.full_like(x, jnp.inf), init_prms)
    clip_max = params_shaper.flatten_single(clip_max)
    mutation_fn = lambda x, k, s: mutate(x, k, cfg.p_duplicate, s.sigma, mutation_mask, params_shaper, clip_min, clip_max, n_types) #type:ignore




    ga = GA(mutation_fn, prms, cfg.pop, elite_ratio=0.5, sigma_init=cfg.sigma, sigma_decay=1., sigma_limit=0.01, p_duplicate=cfg.p_duplicate)

    logger = rx.Logger(True, metrics_fn)
    trainer = rx.EvosaxTrainer(cfg.gens, ga, tsk, params_like=prms, eval_reps=cfg.eval_reps, logger=logger)

    wandb.init(project="evodevox", config=dict())
    s = jax.block_until_ready(trainer.init_and_train_(random_key()))
    x = s.archive[0]
    prms = trainer.params_shaper.reshape_single(x)
    policy = fctry(prms)
    net = policy.initialize(random_key())
    fig = plt.figure()
    ax = fig.add_subplot()
    render_network(net, ax=ax)
    wandb.log({"best network": wandb.Image(fig)})
    wandb.finish()

if __name__ == '__main__':
    jax.config.update("jax_debug_nans", True)

    cfg = Config(pop=256, gens=16, p_duplicate=0.01, sigma=0.01, N0=8)
    train(cfg)

    # tsk = rx.GymnaxTask("CartPole-v1")
    # o, s, *_ = tsk.reset(jr.key(1))
    # k = jr.key(2)
    # for _ in range(10):
    #     k, k1, k2 = jr.split(k, 3)
    #     a = jr.choice(k1, jnp.arange(1))
    #     o, s, *_ = tsk.step(k2, s, a)
    #     print(o)




