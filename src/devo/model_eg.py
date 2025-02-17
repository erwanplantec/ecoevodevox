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
        return jr.multivariate_normal(key, typ.mu, typ.sigma)

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

max_neuron_prms = lambda n_types, n_morphogens, synaptic_markers: NeuronParams(psi=jnp.full(n_morphogens, -jnp.inf), 
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

ZERO = jnp.zeros(())
ONE = jnp.ones(())
    
class Model_EG(eqx.Module):
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
                 migration_T=10.0, migration_dt=0.05, migration_temperature_decay=1., sigma_is_diagonal=False, *, key):

        key_A, key_O = jr.split(key)
        
        self.N = jnp.array([N], dtype=float)
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
        self.types = GaussianType(
            pi = jnp.zeros(max_types).at[0].set(1.),
            mu = jnp.zeros((max_types, n_prms)),
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
    def __call__(self, key):

        k_sample_types, k_sample_prms, k_sample_x, k_migration = jr.split(key, 4)

        # --- 1. Sample Neurons ---
        n_types = self.types.pi.shape[0]
        N = self.N * self.N_gain
        p_active = N / self.N_max
        pi = (self.types.pi / jnp.sum(self.types.pi * self.types.active)) * p_active
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
        gamma = neuron_prms.gamma[:, neuron_types_ids]
        def _step(c, key):
            x, t = c
            x = migration_step(x, t, neuron_prms.psi, gamma, active_neurons, neuron_prms.theta, 
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
        
        return CTRNN(
            a=jnp.zeros(self.N_max), tau=neuron_prms.tau, gain=neuron_prms.gain, 
            bias=neuron_prms.bias, W=W, mask=active_neurons, m=neuron_prms.m, s=neuron_prms.s,
            x=x, id_=neuron_types_ids
        )
    # ---
    def partition(self):
        return eqx.partition(self, eqx.is_array)