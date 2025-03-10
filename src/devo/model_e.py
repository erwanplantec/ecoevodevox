from ..utils.viz import render_network
from .ctrnn import CTRNN, CTRNNPolicy, CTRNNPolicyConfig

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import equinox as eqx
import equinox.nn as nn
import evosax as ex

from typing import Callable, NamedTuple
from jaxtyping import Float, Int


class NeuronType(NamedTuple):
    # ---
    pi: Float
    active: Float
    id_: Int
    # -- Migration Parameters ---
    psi: Float # molecular affinity
    zeta: Float # molecular pertubation
    gamma: Float # repulsion distance decay coefficients
    theta: Float
    # --- Connection Parameters ---
    omega: Float # type-specific gene expression
    # --- Dynamical parameters ---
    tau: Float
    bias: Float
    gain: Float
    s: Float # expression of sensory elements
    m: Float # expression of motor characteristics
    # ---


# ====================== MIGRATION ======================

def migration_field(x):
    return jnp.array([x[0], x[1], jnp.abs(x[0]), jnp.abs(x[1]), jnp.max(jnp.abs(x))])

N_MORPHOGENS = 5

def safe_norm(x):
    return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def migration_step(xs, t, psis, gammas, zetas, mask, thetas, dt=0.01, temperature_decay=0.98, 
    migration_field=migration_field, *, key):
    
    N, _ = xs.shape
    T = temperature_decay ** t

    M = migration_field

    def M_(x):
        """Modified molecular field"""
        d = jnp.sum(jnp.square(x[None]-xs), axis=-1, keepdims=True) #N,1
        return M(x) + jnp.sum(zetas * jnp.exp(-d/(gammas+1e-6)) * mask[:,None], axis=0)

    def E(x, psi):
        """energy field"""
        return jnp.dot(M_(x), psi)

    dx = - jax.vmap(jax.grad(E))(xs, psis)
    dx_norm = jnp.linalg.norm(dx, axis=-1, keepdims=True)
    dx = jnp.where(dx_norm>0, dx/dx_norm, dx)

    xs = jnp.clip(xs + dt * T * dx, -1., 1.)
    
    return xs

# ======================================================


dummy_policy_config = CTRNNPolicyConfig(lambda x:x, lambda x:x)
    
class Model_E(CTRNNPolicy):
    # --- params ---
    types: NeuronType
    N: jax.Array
    O: jax.Array
    A: nn.MLP
    # --- statics ---
    n_types: int
    max_nodes: int
    dt: float
    dvpt_time: float
    temperature_decay: float
    migration_field: Callable
    # ---
    def __init__(self, n_types: int, n_synaptic_markers: int, max_nodes_per_type: int=32, 
                 dt: float=0.1, dvpt_time: float=10., temperature_decay: float=1., extra_migration_fields: int=3,
                 policy_cfg: CTRNNPolicyConfig=dummy_policy_config, *, key: jax.Array):

        super().__init__(policy_cfg)
        
        k1, k2 = jr.split(key)

        n_fields = N_MORPHOGENS + extra_migration_fields
        self.migration_field = lambda x: jnp.concatenate([migration_field(x),jnp.zeros(extra_migration_fields)])
        
        types = NeuronType(
            pi = jnp.zeros(n_types, dtype=jnp.float16),
            psi = jnp.zeros((n_types, n_fields)),
            gamma = jnp.zeros((n_types, n_fields))+0.001,
            zeta = jnp.zeros((n_types, n_fields)),
            omega = jnp.zeros((n_types, n_synaptic_markers)),
            theta = jnp.ones(n_types),
            active = jnp.zeros(n_types),
            id_ = jnp.arange(n_types),
            s = jnp.zeros(n_types),
            m = jnp.zeros(n_types),
            tau = jnp.ones(n_types),
            bias= jnp.zeros(n_types),
            gain = jnp.ones(n_types)
        )
        
        self.types = types
        self.O = jr.normal(k1, (n_synaptic_markers,)*2) * 0.1
        self.A = nn.MLP(n_synaptic_markers+n_fields, n_synaptic_markers, 64, 1, key=k2)
        self.N = jnp.zeros(())
        
        self.n_types = n_types
        self.max_nodes = max_nodes_per_type * n_types
        self.dt = dt
        self.dvpt_time = dvpt_time
        self.temperature_decay = temperature_decay
    # ---
    def initialize(self, key: jax.Array)->CTRNN:
        
        # 1. Initialize neurons
        x0 = jr.normal(key, (self.max_nodes, 2)) * 0.01
        node_type_ids = jnp.zeros(self.max_nodes)
        n_tot = 0
        pi = (self.types.pi*self.types.active) / jnp.sum(self.types.pi * self.types.active)
        n = jnp.round(self.N * 10.0 * pi)
        for _, (n, msk) in enumerate(zip(n, self.types.active)):
            node_type_ids = jnp.where(jnp.arange(self.max_nodes)<n_tot+n*msk, node_type_ids+1, node_type_ids)
            n_tot += n*msk
        node_type_ids = self.n_types - node_type_ids
        node_type_ids = jnp.where(node_type_ids < self.n_types, node_type_ids, -1).astype(int)
        node_types = jax.tree.map(lambda x: x[node_type_ids], self.types)
        
        # 2. Migrate
        step_fn = lambda i, x: migration_step(
            xs=x, t=self.dt*i, psis=node_types.psi, gammas=node_types.gamma, zetas=node_types.zeta, mask=node_types.active, 
            thetas=node_types.theta, dt=self.dt, temperature_decay=self.temperature_decay, migration_field=self.migration_field, 
            key=jr.key(1)
        )
        x = jax.lax.fori_loop(0, int(self.dvpt_time//self.dt), step_fn, x0)
        #x, xs = jax.lax.scan(lambda x, i:(step_fn(i, x),x), x0, jnp.arange(int(self.dvpt_time//self.dt)))
        
        # 3. Connect
        M = jax.vmap(self.migration_field)(x)
        g = jax.vmap(self.A)(jnp.concatenate([node_types.omega,M], axis=-1))
        W = g @ self.O @ g.T
        W = W * (node_types.active[:,None] * node_types.active[None])
        network = CTRNN(
            a=jnp.zeros(x.shape[0]), 
            x=x, W=W, tau=node_types.tau, gain=node_types.gain, bias=node_types.bias, 
            s=node_types.s, m=node_types.m, id_=node_types.id_, mask=node_types.active
        )
        
        return network
    # ---
    def partition(self):
        return eqx.partition(self, eqx.is_array)


# ======================= UTILS ===========================


def make_two_types(mdl, n_sensory_neurons, n_motor_neurons):
    n_synaptic_markers = mdl.types.omega.shape[-1]
    n_total = n_sensory_neurons + n_motor_neurons
    sensory_type = NeuronType(
        pi = n_sensory_neurons/n_total, 
        id_ = 0,
        psi = jnp.array([0.,-1.,0., 0., 0., 1., 0., 0.]),
        gamma = jnp.array([0.,0.,0., 0., 0., 0.05, 0., 0.]),
        zeta = jnp.array([0.,0.,0., 0., 0., 1., 0., 0.]),
        theta = 1.,
        omega = jnn.one_hot(0, n_synaptic_markers),
        active = 1.,
        s = 1.,
        m = 0.,
        tau = 1.,
        bias = 0.,
        gain = 1.
    )
    
    motor_type = NeuronType(
        pi = n_motor_neurons/n_total,
        id_ = 1,
        psi = jnp.array([0.,1.,0., 0., 0., 0., 1., 0.]),
        gamma = jnp.array([0.,0.,0., 0., 0., 0., 0.08, 0.]),
        zeta = jnp.array([0.,0.,0., 0., 0., 0., 1., 0.]),
        theta = 1.,
        omega = jnn.one_hot(1, n_synaptic_markers),
        active = 1.,
        s = 0.,
        m = 1.,
        tau = 1.,
        bias = 0.,
        gain = 1.
    )
    
    types = mdl.types
    types = jax.tree.map(lambda x, y: x.at[0].set(y), types, sensory_type)
    types = jax.tree.map(lambda x, y: x.at[1].set(y), types, motor_type)
    mdl = eqx.tree_at(lambda x: [x.types, x.N], mdl, [types, n_total/10.])
    return mdl

def make_single_type(mdl, n_neurons):
    n_synaptic_markers = mdl.types.omega.shape[-1]
    sensorimotor_type = NeuronType(
        pi = 1., 
        id_ = 0,
        psi = jnp.array([0.,0.,0., 0., 0., 1., 0., 0.]),
        gamma = jnp.array([0.,0.,0., 0., 0., 0.1, 0., 0.]),
        zeta = jnp.array([0.,0.,0., 0., 0., 1., 0., 0.]),
        theta = 1.,
        omega = jnn.one_hot(0, n_synaptic_markers),
        active = 1.,
        s = 1.,
        m = 1.,
        tau = 1.,
        bias = 0.,
        gain = 1.
    )
    
    types = mdl.types
    types = jax.tree.map(lambda x, y: x.at[0].set(y), types, sensorimotor_type)
    mdl = eqx.tree_at(lambda x: [x.types,x.N], mdl, [types,n_neurons/10.])
    return mdl

# ========================= EVOLUTION =============================

def duplicate_type(model, key):
    k1, k2 = jr.split(key)
    types = model.types
    k = model.types.active.sum().astype(int)
    p = jnp.arange(types.psi.shape[0])<k
    p = p / p.sum()
    i = jr.choice(k1, jnp.arange(types.psi.shape[0]), p=p)
    
    copied_type = jax.tree.map(lambda x:x[i], types)
    types = jax.tree.map(lambda t, ct: t.at[k].set(ct), types, copied_type)
    
    p = jr.uniform(k2)
    pi_i = p * types.pi[i]
    pi_k = (1-p) * types.pi[i]
    pi = types.pi.at[k].set(pi_k)
    pi = pi.at[i].set(pi_i)
    types = eqx.tree_at(lambda types: types.pi, types, pi)
    
    types = eqx.tree_at(lambda types: types.id_, types, types.id_.at[k].set(k))
    model = eqx.tree_at(lambda m: m.types, model, types)
    
    return model, jnn.one_hot(i, num_classes=types.psi.shape[0])


def mutate(prms: jax.Array, key: jax.Array, p_duplicate: float, sigma_mut: float, 
           mutation_mask: jax.Array, shaper: ex.ParameterReshaper, 
           clip_min: jax.Array, clip_max: jax.Array, n_types: int):

    def _duplicate(prms, key):
        mdl = shaper.reshape_single(prms)
        mdl, dupl = duplicate_type(mdl, key)
        prms = shaper.flatten_single(mdl)
        return prms, dupl

    def _mutate(prms, key):
        epsilon = jr.normal(key, prms.shape) * sigma_mut * mutation_mask
        prms = prms + epsilon
        prms = jnp.clip(prms, clip_min, clip_max)
        return prms

    k1, k2 = jr.split(key)
    prms, duplicated = jax.lax.cond(
        jr.uniform(k1)<p_duplicate,
        lambda prms, key: _duplicate(prms, key),
        lambda prms, key: (_mutate(prms, key), jnp.zeros(n_types,)),
        prms, k2
    )
    
    return prms, duplicated


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    model = Model_E(4, N_MORPHOGENS, 8, key=jr.key(1))
    model = make_two_types(model, 10, 4)
    ctrnn = model.initialize(jr.key(2))

    # fig, ax = plt.subplots(1,2, figsize=(10,5), sharey=True)
    # msk = ctrnn.mask.astype(bool)
    # c = plt.cm.Set1(ctrnn.id_[msk])
    # for i, x in enumerate(xs[:-1]):
    #     alpha = (i/xs.shape[0]) * 0.5 + 0.1
    #     ax[0].scatter(*x[msk].T, c=c, alpha=alpha)

    # x = xs[-1,msk]

    # render_network(ctrnn, ax=ax[1])

    # plt.show()
    # 