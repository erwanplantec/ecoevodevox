import matplotlib
from .policy import CTRNNPolicy, CTRNNPolicyConfig, CTRNN
from .model_e import migration_step, morphogen_field, N_MORPHOGENS

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import equinox as eqx
import equinox.nn as nn
import evosax as ex
from typing import Callable, NamedTuple, TypeAlias
from jaxtyping import Float

ZERO = jnp.zeros(())
ONE = jnp.ones(())


class Decoder(eqx.Module):
    # ---
    mlp: nn.MLP
    # ---
    def __init__(self, latent_dims: int, output_dims: int, width=32, depth=2, *, key: jax.Array, **kwargs):
        self.mlp = nn.MLP(latent_dims, output_dims, width, depth, **kwargs, key=key)
    # ---
    def __call__(self, z: jax.Array):
        return self.mlp(z)

class NeuronType(NamedTuple):
    pi: Float
    z: jax.Array
    active: Float

class NeuronParams(NamedTuple):
    # --- Migration ---
    psi: Float
    gamma: Float
    theta: Float
    alpha: Float
    beta: Float
    # --- Synapses ---
    omega: Float
    # --- Dynamics ---
    gain: Float
    tau: Float
    bias: Float
    m: Float
    s: Float



min_neuron_prms = lambda n_types, n_morphogens, synaptic_markers: NeuronParams(psi=jnp.full(n_morphogens, -jnp.inf), 
                                                                               gamma=ZERO,
                                                                               theta=ZERO,
                                                                               alpha=ZERO,
                                                                               beta=ZERO,
                                                                               omega=jnp.full(synaptic_markers, -jnp.inf),
                                                                               gain=ZERO,
                                                                               tau=0.01,
                                                                               bias=-jnp.inf,
                                                                               m=-ONE,
                                                                               s=-ONE)

max_neuron_prms = lambda n_types, n_morphogens, synaptic_markers: NeuronParams(psi=jnp.full(n_morphogens, jnp.inf), 
                                                                               gamma=jnp.inf,
                                                                               theta=jnp.inf,
                                                                               alpha=ONE,
                                                                               beta=ONE,
                                                                               omega=jnp.full(synaptic_markers, jnp.inf),
                                                                               gain=jnp.inf,
                                                                               tau=10.,
                                                                               bias=jnp.inf,
                                                                               m=ONE,
                                                                               s=ONE)

sensorimotor_neuron = lambda n_types, n_morphogens, synaptic_markers : NeuronParams(psi=jnp.zeros(n_morphogens), 
                                                                                    gamma=0.5,
                                                                                    theta=0.,
                                                                                    alpha=1.,
                                                                                    beta=0.2,
                                                                                    omega=jnp.zeros(synaptic_markers).at[0].set(1.),
                                                                                    gain=1.,
                                                                                    tau=1.,
                                                                                    bias=0.,
                                                                                    m=1.,
                                                                                    s=1.)


dummy_policy_config = CTRNNPolicyConfig(lambda x:x, lambda x:x)

class Model_LG(CTRNNPolicy):
    # --- params ---
    types: NeuronType
    type_decoder: Decoder
    N: jax.Array
    O: jax.Array
    A: nn.MLP
    neuron_prms_bias: jax.Array
    # --- statics ---
    alpha: float
    beta: float
    N_gain: float
    N_max: int
    migration_iters: int
    migration_dt: float
    migration_temperature_decay: float
    _neuron_prms_shaper: Callable
    _clip_neuron_prms: Callable
    # ---
    def __init__(self, latent_dims=16, N=8, N_max=256, max_types=16, N_gain=10.0, synaptic_markers=8, n_morphogens=N_MORPHOGENS, 
                 migration_T=10.0, migration_dt=0.05, migration_temperature_decay=1., 
                 alpha=1., beta=0.5, policy_config: CTRNNPolicyConfig=dummy_policy_config, *, key):
        
        super().__init__(policy_config)

        key_dec, key_A, key_O = jr.split(key, 3)

        self.N = jnp.array(N, dtype=float) / N_gain
        self.N_gain = N_gain
        self.N_max = N_max

        self.types = NeuronType(
            pi = jnp.zeros(max_types),
            z = jnp.zeros((max_types, latent_dims)),
            active = jnp.zeros(max_types).at[0].set(1.)
        )

        _min_neuron_prms = min_neuron_prms(max_types, n_morphogens, synaptic_markers)
        _max_neuron_prms = max_neuron_prms(max_types, n_morphogens, synaptic_markers)
        flat_min_neuron_prms, shaper = jax.flatten_util.ravel_pytree(_min_neuron_prms) #type:ignore
        flat_max_neuron_prms, shaper = jax.flatten_util.ravel_pytree(_max_neuron_prms) #type:ignore
        self._neuron_prms_shaper = shaper
        self.type_decoder = Decoder(latent_dims, len(flat_min_neuron_prms), key=key_dec)
        self._clip_neuron_prms = lambda prms: jnp.clip(prms, flat_min_neuron_prms, flat_max_neuron_prms)
        self.alpha = alpha
        self.beta = beta

        self.migration_dt = migration_dt
        self.migration_iters = int(migration_T // migration_dt)
        self.migration_temperature_decay = migration_temperature_decay

        self.A = nn.MLP(n_morphogens+synaptic_markers, synaptic_markers, 16, 1, key=key_A, final_activation=jnn.tanh)
        self.O = jr.normal(key_O, (synaptic_markers, synaptic_markers))

        self.neuron_prms_bias, _ = jax.flatten_util.ravel_pytree(sensorimotor_neuron(max_types, n_morphogens, synaptic_markers)) #type:ignore
    
    # --- 

    def initialize(self, key: jax.Array) -> CTRNN:
        
        n_types = self.types.pi.shape[0]
        k_sample_types, k_sample_x, k_migration = jr.split(key, 3)

        # --- 1. Sample Neurons ---
        N = jnp.clip(self.N * self.N_gain, 0., self.N_max)
        p_active = N  / self.N_max
        pi = self.types.pi * self.types.active
        pi = (pi / jnp.sum(pi)) * p_active
        pi = jnp.concatenate([pi, jnp.array([1-p_active])])

        neuron_types_ids = jr.choice(k_sample_types, jnp.arange(n_types+1).at[-1].set(-1), (self.N_max,), p=pi)
        neuron_types = jax.tree.map(lambda x: x[neuron_types_ids], self.types)
        neuron_prms = jax.vmap(self.type_decoder)(neuron_types.z)
        neuron_prms = neuron_prms + self.neuron_prms_bias[None]
        neuron_prms = jax.vmap(self._clip_neuron_prms)(neuron_prms)
        neuron_prms = jax.vmap(self._neuron_prms_shaper)(neuron_prms)
        active_neurons = jnp.where(neuron_types_ids==-1, 0., 1.)

        # --- 2. Migrate ---
        x = jr.normal(k_sample_x, (self.N_max, 2)) * 0.01
        def _step(c, key):
            x, t = c
            x = migration_step(x, t, neuron_prms.psi, neuron_prms.gamma, active_neurons, neuron_prms.theta, 
                               dt=self.migration_dt, alpha=neuron_prms.alpha[:,None], beta=neuron_prms.beta[:,None], 
                               temperature_decay=self.migration_temperature_decay, key=key)
            return [x, t+self.migration_dt], x
        [x, _], _ = jax.lax.scan(
            _step, [x, 0.0], jr.split(k_migration, self.migration_iters)
        )

        # --- 3. Connect Neurons ---
        M = jax.vmap(morphogen_field)(x)
        S = jax.vmap(self.A)(jnp.concatenate([neuron_prms.omega, M], axis=-1))
        W = S @ self.O @ S.T
        W = W * (active_neurons[None]*active_neurons[:,None])

        return CTRNN(
            a=jnp.zeros(self.N_max), tau=neuron_prms.tau, gain=neuron_prms.gain, 
            bias=neuron_prms.bias, W=W, mask=active_neurons, m=neuron_prms.m, s=neuron_prms.s,
            x=x, id_=neuron_types_ids
        )
    # ---
    def partition(self):
        return eqx.partition(self, eqx.is_array)
    # ---
    def prms_lower_bound(self):
        prms, _ = self.partition()
        prms = jax.tree.map(lambda x: jnp.full_like(x, -jnp.inf), prms)
        prms = eqx.tree_at(
            lambda tree: [tree.N, tree.types.pi],
            prms,
            [jnp.zeros_like(prms.N), jnp.zeros_like(prms.types.pi)]
        )
        return prms
    # ---
    def prms_upper_bound(self):
        prms, _ = self.partition()
        prms = jax.tree.map(lambda x: jnp.full_like(x, jnp.inf), prms)
        prms = eqx.tree_at(
            lambda tree: [tree.N],
            prms,
            [jnp.full_like(prms.N, self.N_max/self.N_gain)]
        )
        return prms

def mutate(prms: jax.Array, key: jax.Array, p_duplicate: float, sigma: float, shaper: ex.ParameterReshaper, mask: jax.Array, clip_fn: Callable):

    def _duplicate(prms, key):
        k1, k2 = jr.split(key)
        mdl = shaper.reshape_single(prms)
        n_types = mdl.types.pi.shape[0]
        k = mdl.types.active.sum().astype(int)
        i = jr.choice(k1, jnp.arange(n_types))
        mdl_types = jax.tree.map(lambda x: x.at[k].set(x[i]), mdl.types)
        
        r = jr.uniform(k2)
        pi_i = mdl_types.pi[i]
        pi = mdl_types.pi.at[i].set(pi_i*r).at[k].set(pi_i*(1-r))

        active = mdl_types.active.at[k].set(1.)
        mdl_types = eqx.tree_at(lambda x: [x.pi,x.active], mdl_types, [pi, active])
        mdl = eqx.tree_at(lambda t: t.types, mdl, mdl_types)

        return shaper.flatten_single(mdl)

    k1, k2 = jr.split(key)
    prms = jax.lax.cond(
        jr.uniform(k1) < p_duplicate,
        _duplicate,
        lambda prms, key: prms,
        prms, k2
    )
    
    epsilon = jr.normal(key, prms.shape) * sigma * mask
    prms = clip_fn(prms + epsilon)

    return prms



if __name__ == '__main__':
    import random
    import matplotlib.pyplot as plt
    import evosax as ex
    from ..utils.viz import render_network

    plt.style.use("../ntbks/drk_fte.mplstyle")
    key = jr.key(random.randint(0, 1000))
    fig, ax = plt.subplots(2,5, figsize=(20,8), sharey="row")
    model = Model_LG(N=8, N_max=32, max_types=8, key=key, beta=1., alpha=1.)
    prms, _ = model.partition()
    shaper = ex.ParameterReshaper(prms, verbose=True)
    mut_msk = jax.tree.map(lambda x: jnp.ones_like(x), prms)
    mut_msk = eqx.tree_at(lambda tree: tree.types.active, mut_msk, jnp.zeros_like(prms.types.active))
    mut_msk = shaper.flatten_single(prms)
    lower_bound, upper_bound = model.prms_lower_bound(), model.prms_upper_bound()
    lower_bound = shaper.flatten_single(lower_bound)
    upper_bound = shaper.flatten_single(upper_bound)
    clip_fn = lambda prms: jnp.clip(prms, lower_bound, upper_bound)

    ctrnn = model.initialize(key)
    wmax = jnp.abs(ctrnn.W.max())
    render_network(ctrnn, ax=ax[0,0])
    sc = ax[1,0].imshow(ctrnn.W, cmap="coolwarm", vmin=-wmax, vmax=wmax)
    plt.colorbar(sc)
    for i in range(4):
        key, _key = jr.split(key)
        prms = shaper.flatten_single(prms)
        prms = mutate(prms, _key, 0.1, 0.1, shaper, mut_msk, clip_fn)
        model = eqx.combine(shaper.reshape_single(prms), _)
        pi = model.types.pi * model.types.active
        ctrnn = model.initialize(_key)
        render_network(ctrnn, ax=ax[0, i+1])
        wmax = jnp.abs(ctrnn.W.max())
        sc = ax[1,i+1].imshow(ctrnn.W, cmap="coolwarm", vmin=-wmax, vmax=wmax)
        plt.colorbar(sc)
    fig.tight_layout()
    plt.show()













