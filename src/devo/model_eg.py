
from ..utils.viz import render_network, plt
from .model_e import morphogen_field, N_MORPHOGENS, migration_step
from .policy import CTRNN, CTRNNPolicy, CTRNNPolicyConfig

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import equinox as eqx
import equinox.nn as nn
import evosax as ex

from typing import NamedTuple, Callable
from jaxtyping import Float, Int

class GaussianType(eqx.Module):
    # ---
    pi: jax.Array
    mu: jax.Array
    sigma: jax.Array
    active: jax.Array
    # ---

def sample_type(typ: GaussianType, key):
    if typ.sigma.ndim==1:
        return jr.normal(key, typ.mu.shape) * typ.sigma[:,None] + typ.mu
    else:
        return jr.multivariate_normal(key, typ.mu, typ.sigma, method="svd")

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

sensory_neuron = lambda n_types, n_morphogens, synaptic_markers : NeuronParams(psi=jnp.zeros(n_morphogens), 
                                                                               gamma=0.,
                                                                               theta=0.,
                                                                               alpha=1.,
                                                                               beta=0.2,
                                                                               omega=jnp.zeros(synaptic_markers).at[0].set(1.),
                                                                               gain=1.,
                                                                               tau=1.,
                                                                               bias=0.,
                                                                               m=0.,
                                                                               s=1.)

motor_neuron = lambda n_types, n_morphogens, synaptic_markers : NeuronParams(psi=jnp.zeros(n_morphogens), 
                                                                             gamma=0.,
                                                                             theta=0.,
                                                                             alpha=1.,
                                                                             beta=0.2,
                                                                             omega=jnp.zeros(synaptic_markers).at[0].set(1.),
                                                                             gain=1.,
                                                                             tau=1.,
                                                                             bias=0.,
                                                                             m=1.,
                                                                             s=0.)


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

ZERO = jnp.zeros(())
ONE = jnp.ones(())

dummy_policy_config = CTRNNPolicyConfig(lambda x:x, lambda x: x)
    
class Model_EG(CTRNNPolicy):
    # ---
    N: jax.Array
    O: jax.Array
    A: nn.MLP
    types: GaussianType
    # ---
    N_gain: float
    N_max: int
    migration_iters: int
    migration_dt: float
    migration_temperature_decay: float
    _prms_shaper: Callable
    _clip_prms: Callable
    # ---
    def __init__(self, N=8, N_max=256, max_types=16, N_gain=10.0, synaptic_markers=8, n_morphogens=N_MORPHOGENS, 
                 migration_T=10.0, migration_dt=0.05, migration_temperature_decay=1., sigma_is_diagonal=False, 
                 policy_config: CTRNNPolicyConfig=dummy_policy_config, *, key):


        super().__init__(policy_config)

        key_A, key_O = jr.split(key)
        
        self.N = jnp.array(N, dtype=float) / N_gain
        self.N_gain = N_gain
        self.N_max = N_max

        dummy_prms = NeuronParams(jnp.zeros(n_morphogens), ZERO, 
                                  ZERO, ZERO, ZERO, jnp.zeros(synaptic_markers), ZERO, 
                                  ZERO, ZERO, ZERO, ZERO)
        flat_dummy_prms, self._prms_shaper = jax.flatten_util.ravel_pytree(dummy_prms) #type:ignore
        n_prms = len(flat_dummy_prms)

        if not sigma_is_diagonal:
            sigma = jnp.stack([jnp.identity(n_prms)*0.1 for _ in range(max_types)])
        else:
            sigma = jnp.full((max_types, n_prms), 0.1)

        sensorimotor_prms, _ = jax.flatten_util.ravel_pytree(sensorimotor_neuron(max_types, n_morphogens, synaptic_markers))#type:ignore
        self.types = GaussianType(
            pi = jnp.zeros(max_types).at[0].set(1.),
            mu = jnp.zeros((max_types, n_prms)).at[0].set(sensorimotor_prms),
            sigma = sigma,
            active = jnp.zeros(max_types).at[0].set(1.)
        )

        self.migration_dt = migration_dt
        self.migration_iters = int(migration_T // migration_dt)
        self.migration_temperature_decay = migration_temperature_decay

        self.A = nn.MLP(n_morphogens+synaptic_markers, synaptic_markers, 16, 1, key=key_A, final_activation=jnn.tanh)
        self.O = jr.normal(key_O, (synaptic_markers, synaptic_markers))

        max_prms = max_neuron_prms(max_types, n_morphogens, synaptic_markers)
        clip_max, _ = jax.flatten_util.ravel_pytree(max_prms) #type:ignore
        min_prms = min_neuron_prms(max_types, n_morphogens, synaptic_markers)
        clip_min, _ = jax.flatten_util.ravel_pytree(min_prms) #type:ignore
        self._clip_prms = lambda prms: jnp.clip(prms, clip_min, clip_max)
    # ---
    def initialize(self, key):

        k_sample_types, k_sample_prms, k_sample_x, k_migration = jr.split(key, 4)

        # --- 1. Sample Neurons ---
        n_types = self.types.pi.shape[0]
        N = jnp.clip(self.N * self.N_gain, 0., self.N_max)
        p_active = N  / self.N_max
        pi = self.types.pi * self.types.active
        pi = (pi / jnp.sum(pi)) * p_active
        pi = jnp.concatenate([pi, jnp.array([1-p_active])])

        # Use -1 as a dummy index indicating unused neurons
        neuron_types_ids = jr.choice(k_sample_types, jnp.arange(n_types+1).at[-1].set(-1), (self.N_max,), p=pi)
        neuron_prms_gaussian = jax.tree.map(lambda x: x[neuron_types_ids], self.types)
        neuron_prms = jax.vmap(sample_type)(neuron_prms_gaussian, jr.split(k_sample_prms, self.N_max)) 
        neuron_prms = jax.vmap(self._clip_prms)(neuron_prms)
        neuron_prms = jax.vmap(self._prms_shaper)(neuron_prms)
        active_neurons= jnp.where(neuron_types_ids==-1, 0., 1.)

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


def mutate(key: jax.Array, prms: jax.Array, sigma: float, p_duplicate: float, shaper: ex.ParameterReshaper, n_types: int, synaptic_markers: int, n_morphogens: int):
    # ---
    def _duplicate(prms, key):
        k1, k2 = jr.split(key)
        mdl = shaper.reshape_single(prms)
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
    # ---
    k1, k2, k3 = jr.split(key, 3)
    prms = jax.lax.cond(
        jr.uniform(k1) < p_duplicate,
        _duplicate,
        lambda prms, key: prms,
        prms, k2
    )
    # ---
    neuron_prms_min, _ = jax.flatten_util.ravel_pytree(min_neuron_prms(n_types, n_morphogens, synaptic_markers)) #type:ignore
    neuron_prms_max, _ = jax.flatten_util.ravel_pytree(max_neuron_prms(n_types, n_morphogens, synaptic_markers)) #type:ignore
    
    prms_min = jax.tree.map(lambda x: jnp.full_like(x, -jnp.inf), shaper.reshape_single(prms))
    gaussian_mins = eqx.tree_at(lambda x: [x.mu, x.sigma, x.pi], 
                                prms_min.types, 
                                [
                                    neuron_prms_min[None].repeat(n_types,0), 
                                    jnp.full_like(prms_min.types.sigma, 0.01), 
                                    jnp.full_like(prms_min.types.pi, 0.)
                                ]
   )
    prms_min = eqx.tree_at(lambda x: x.types, prms_min, gaussian_mins)

    prms_max = jax.tree.map(lambda x: jnp.full_like(x, jnp.inf), shaper.reshape_single(prms))
    gaussian_max = eqx.tree_at(lambda x: [x.mu, x.sigma, x.pi], 
                                prms_max.types, 
                                [
                                    neuron_prms_max.repeat(n_types,0), 
                                    jnp.full_like(prms_max.types.sigma, jnp.inf), 
                                    jnp.full_like(prms_min.types.pi, jnp.inf)])
    prms_max = eqx.tree_at(lambda x: x.types, prms_max, gaussian_max)

    mask, _ = jax.flatten_util.ravel_pytree( #type:ignore
        eqx.tree_at(lambda x: x.types.active, jax.tree.map(lambda x: jnp.ones_like(x), prms_max), jnp.zeros_like(prms_max.types.active))
    )
    prms_max, _ = jax.flatten_util.ravel_pytree(prms_max) #type:ignore
    prms_min, _ = jax.flatten_util.ravel_pytree(prms_min) #type:ignore

    # ---
    epsilon = jr.normal(k3, prms.shape) * sigma * mask
    prms = prms + epsilon
    prms = jnp.clip(prms, prms_min, prms_max)

    return prms

if __name__ == '__main__':
    import random
    plt.style.use("../ntbks/drk_fte.mplstyle")
    key = jr.key(random.randint(0, 1000))
    fig, ax = plt.subplots(2,5, figsize=(20,8), sharey="row")
    model = Model_EG(N=8, N_max=32, max_types=8, key=key, sigma_is_diagonal=False,)
    ctrnn = model.initialize(key)
    wmax = jnp.abs(ctrnn.W.max())
    render_network(ctrnn, ax=ax[0,0])
    sc = ax[1,0].imshow(ctrnn.W, cmap="coolwarm", vmin=-wmax, vmax=wmax)
    plt.colorbar(sc)
    for i in range(4):
        key, _key = jr.split(key)
        prms, _ = model.partition()
        shaper = ex.ParameterReshaper(prms, verbose=False)
        prms_flat = shaper.flatten_single(prms)
        prms = mutate(_key, prms_flat, sigma=0.1, p_duplicate=0.5, shaper=ex.ParameterReshaper(prms, verbose=False), n_types=8, synaptic_markers=8, n_morphogens=7)
        model = eqx.combine(shaper.reshape_single(prms), _)
        print(model.types.active)
        print(ctrnn.id_)
        pi = model.types.pi *model.types.active
        print(pi / pi.sum())
        print("# ---")
        ctrnn = model.initialize(_key)
        render_network(ctrnn, ax=ax[0, i+1])
        wmax = jnp.abs(ctrnn.W.max())
        sc = ax[1,i+1].imshow(ctrnn.W, cmap="coolwarm", vmin=-wmax, vmax=wmax)
        plt.colorbar(sc)
    fig.tight_layout()
    plt.show()






































