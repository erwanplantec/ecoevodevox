from chex import ArrayTree
from jax.flatten_util import ravel_pytree
from .base import BaseDevelopmentalModel
from .policy_network.ctrnn import SECTRNN

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import equinox as eqx
import equinox.nn as nn
import evosax as ex

from typing import Any, Callable
from flax.struct import PyTreeNode
from jaxtyping import (
    PyTree,
    Bool,
    Float16, Float32, 
    Int8, Int16, Int32,
    UInt8, UInt16, UInt32)

class TypeBasedSECTRNN(SECTRNN):
    """Type Based Spatially Embedded CTRNN"""
    s: jax.Array
    m: jax.Array
    id_: jax.Array


class NeuronType(PyTreeNode):
    # ---
    pi:     Float32|Float16
    active: Bool
    # -- Migration Parameters ---
    psi:    Float32|Float16 # molecular affinity
    zeta:   Float32|Float16 # molecular pertubation
    gamma:  Float32|Float16 # repulsion distance decay coefficients
    theta:  Float32|Float16
    # --- Connection Parameters ---
    omega:  Float32|Float16 # type-specific gene expression
    # --- Dynamical parameters ---
    tau:    Float32|Float16
    bias:   Float32|Float16
    gain:   Float32|Float16
    s:      Float32|Float16 # expression of sensory elements
    m:      Float32|Float16 # expression of motor characteristics
    # ---
    id_:    UInt8|None=None


# ====================== MIGRATION ======================

def migration_field(x):
    return jnp.array([x[0], x[1], jnp.abs(x[0]), jnp.abs(x[1]), jnp.max(jnp.abs(x))])

N_MORPHOGENS = 5

def safe_norm(x):
    return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)

def migration_step(xs, t, psis, gammas, zetas, mask, thetas, dt=0.01, temperature_decay=0.98, 
    migration_field=migration_field, T_interactions=0.0, shape: str="square", *, key):
    
    N, _ = xs.shape
    T = temperature_decay ** t
    k = jnn.sigmoid((t - T_interactions)*10.0)
    
    thetas_l, thetas_u = thetas.T
    mask = mask & (t>thetas_l) & (t<thetas_u)

    M = migration_field
    def M_(x):
        """Modified molecular field"""
        d = jnp.sum(jnp.square(x[None]-xs), axis=-1, keepdims=True) #N,1
        perturbations = zetas * jnp.exp(-d/(gammas+1e-6)) #N, M
        perturbations = jnp.where(mask[:,None], perturbations, jnp.zeros_like(perturbations)) #N, M
        perturbation = k * jnp.sum(perturbations, axis=0) #type:ignore
        return M(x) + perturbation

    def E(x, psi):
        """energy field"""
        return jnp.dot(M_(x), psi)

    dx = - jax.vmap(jax.grad(E))(xs, psis)
    dx_norm = jnp.linalg.norm(dx, axis=-1, keepdims=True)
    dx = jnp.where(dx_norm>0, dx/dx_norm, dx)
    dx = jnp.where(mask[:,None], dx, jnp.zeros_like(dx))

    if shape=="square":
        xs = jnp.clip(xs + dt * T * dx, -1., 1.)
    elif shape=="circle":
        norms = jnp.linalg.norm(xs, axis=-1, keepdims=True)
        xs = jnp.where(norms>1.0, xs/norms, xs)
    else:
        raise ValueError(f"shape {shape} is not valid. Must be either 'circle' or 'square'")
    
    return xs

# ======================================================

class XOXT(eqx.Module):
    O: jax.Array
    bound: float|None
    def __init__(self, dims, bound=-10.0, *, key: jax.Array):
        self.O = jnp.zeros((dims,dims))
        self.bound=bound
    def __call__(self, x_pre, x_post):
        return jnn.tanh(x_pre @ self.O @ x_post)*self.bound

class MLPConn(nn.MLP):
    def __init__(self, dims, key):
        super().__init__(dims*2, "scalar", 16, 1, key=key)
    def __call__(self, x_pre, x_post): #type:ignore
        return super().__call__(jnp.concatenate([x_pre,x_post]))

# ======================================================

    
class Model_E(BaseDevelopmentalModel):
    # --- params ---
    types: NeuronType
    connection_model: PyTree
    synaptic_expression_model: nn.MLP
    # --- statics ---
    n_types: int
    max_nodes: int
    dt: float
    dvpt_time: float
    temperature_decay: float
    migration_field: Callable
    N_gain: float
    body_shape: str
    # ---
    def __init__(self, 
        n_types: int, 
        n_synaptic_markers: int, 
        max_nodes: int=32, 
        sensory_dimensions: int=1, 
        motor_dimensions: int=1, 
        dt: float=0.1, 
        dvpt_time: float=10., 
        temperature_decay: float=1., 
        extra_migration_fields: int=3,
        N_gain: float=10.0,  
        body_shape: str="square", 
        connection_model: str="xoxt", 
        *,
        key: jax.Array):
        
        k1, k2 = jr.split(key)

        n_fields = N_MORPHOGENS + extra_migration_fields
        self.migration_field = lambda x: jnp.concatenate([migration_field(x),jnp.zeros(extra_migration_fields)])
        
        types = NeuronType(
            pi     = jnp.zeros(n_types),
            psi    = jnp.zeros((n_types, n_fields)),
            gamma  = jnp.zeros((n_types, n_fields))+0.001,
            zeta   = jnp.zeros((n_types, n_fields)),
            omega  = jnp.zeros((n_types, n_synaptic_markers)),
            theta  = jnp.ones((n_types,2)).at[:,0].set(0.01),
            active = jnp.zeros(n_types, dtype=bool).at[0].set(True),
            s      = jnp.zeros((n_types,sensory_dimensions)),
            m      = jnp.zeros((n_types, motor_dimensions)),
            tau    = jnp.ones(n_types),
            bias   = jnp.zeros(n_types),
            gain   = jnp.ones(n_types)
        )
        
        self.types = types
        if connection_model=="xoxt":
            self.connection_model = XOXT(n_synaptic_markers, key=k1)
        elif connection_model=="mlp":
            self.connection_model = MLPConn(n_synaptic_markers, k1)
        else:
            raise ValueError("no such conn model")

        self.synaptic_expression_model = nn.MLP(n_synaptic_markers+n_fields, n_synaptic_markers, 32, 1, key=k2, activation=jnn.sigmoid)
        
        self.n_types = n_types
        self.max_nodes = max_nodes
        self.dt = dt
        self.dvpt_time = dvpt_time
        self.temperature_decay = temperature_decay
        self.N_gain = N_gain
        self.body_shape = body_shape
    # ---
    def __call__(self, key: jax.Array)->TypeBasedSECTRNN:
        
        # 1. Initialize neurons
        x0 = jr.normal(key, (self.max_nodes, 2)) * 0.1
        node_type_ids = jnp.zeros(self.max_nodes, dtype=jnp.uint8)
        n_tot = 0
        pi = self.types.pi * self.types.active
        ns = jnp.ceil(pi * self.N_gain)
        for _, (n, msk) in enumerate(zip(ns, self.types.active)):
            node_type_ids = jnp.where(jnp.arange(self.max_nodes)<n_tot+n*msk, node_type_ids+1, node_type_ids)
            n_tot += n*msk
        node_type_ids = self.n_types - node_type_ids
        node_type_ids = jnp.where(node_type_ids < self.n_types, node_type_ids, -1).astype(jnp.uint8)
        node_types = jax.tree.map(
            lambda x: x[node_type_ids], 
            eqx.tree_at(lambda t:t.id_, self.types, jnp.arange(self.n_types, dtype=jnp.uint8))
        )
        
        # 2. Migrate
        step_fn = lambda i, x: migration_step(
            xs=x, t=self.dt*i, psis=node_types.psi, gammas=node_types.gamma, zetas=node_types.zeta, mask=node_types.active, 
            thetas=node_types.theta*self.dvpt_time, dt=self.dt, temperature_decay=self.temperature_decay, migration_field=self.migration_field, 
            shape=self.body_shape, key=jr.key(1)
        )
        xs = jax.lax.fori_loop(0, int(self.dvpt_time//self.dt), step_fn, x0)
        
        # 3. Connect
        def molecular_field(x):
            """Modified molecular field"""
            d = jnp.sum(jnp.square(x[None]-xs), axis=-1, keepdims=True) #N,1
            return self.migration_field(x) + jnp.sum(node_types.zeta * jnp.exp(-d/(node_types.gamma+1e-6)) * node_types.active[:,None], axis=0)
        
        M = jax.vmap(molecular_field)(xs)
        g = jax.vmap(self.synaptic_expression_model)(jnp.concatenate([node_types.omega,M], axis=-1)) #n,2s+2
        W = jax.vmap(jax.vmap(self.connection_model, in_axes=(0,None)), in_axes=(None,0))(g,g)
        W = W * (node_types.active[:,None] * node_types.active[None])
        
        network = TypeBasedSECTRNN(
            v    = jnp.zeros(xs.shape[0]), 
            x    = xs, 
            W    = W, 
            tau  = node_types.tau, 
            gain = node_types.gain, 
            bias = node_types.bias, 
            s    = node_types.s, 
            m    = node_types.m, 
            id_  = node_types.id_, 
            mask = node_types.active
        )
        
        return network
    # ---
    def partition(self):
        return eqx.partition(self, eqx.is_array)


# =================== HANDCRAFTED NETWORKS ==========================


def make_two_types(mdl, n_sensory_neurons, n_motor_neurons):
    n_synaptic_markers = mdl.types.omega.shape[-1]

    sensory_type = NeuronType(
        pi = n_sensory_neurons/mdl.N_gain, 
        psi = jnp.array([0.,-1.,0., 0., 0., 1., 0., 0.]),
        gamma = jnp.array([0.,0.,0., 0., 0., 0.05, 0., 0.]),
        zeta = jnp.array([0.,0.,0., 0., 0., 1., 0., 0.]),
        theta = jnp.array([0.01, 1.0]),
        omega = jnn.one_hot(0, n_synaptic_markers),
        active = 1.,
        s = 1.,
        m = 0.,
        tau = 1.,
        bias = 0.,
        gain = 1.
    )
    
    motor_type = NeuronType(
        pi = n_motor_neurons/mdl.N_gain,
        psi = jnp.array([0.,1.,0., 0., 0., 0., 1., 0.]),
        gamma = jnp.array([0.,0.,0., 0., 0., 0., 0.08, 0.]),
        zeta = jnp.array([0.,0.,0., 0., 0., 0., 1., 0.]),
        theta = jnp.array([0.01, 1.0]),
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
    mdl = eqx.tree_at(lambda x: [x.types], mdl, [types])
    return mdl

def make_single_type(mdl, n_neurons):
    n_synaptic_markers = mdl.types.omega.shape[-1]
    sensorimotor_type = NeuronType(
        pi = n_neurons/mdl.N_gain, 
        psi = jnp.array([0.,0.,0., 0., 0., 1., 0., 0.]),
        gamma = jnp.array([0.,0.,0., 0., 0., 0.1, 0., 0.]),
        zeta = jnp.array([0.,0.,0., 0., 0., 1., 0., 0.]),
        theta = jnp.array([0.01, 1.0]),
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
    mdl = eqx.tree_at(lambda x: [x.types], mdl, [types])
    return mdl


# ========================= EVOLUTION =============================

def duplicate_type(model: Model_E, key, split_pop=True):
    k1, k2 = jr.split(key)
    types = model.types
    n_types = model.types.pi.shape[0]
    k = model.types.active.sum().astype(int)
    p = (jnp.arange(n_types)<k).astype(float)
    p = p / p.sum()
    i = jr.choice(k1, jnp.arange(n_types), p=p)
    
    types = jax.tree.map(lambda x: x.at[k].set(x[i]), types)
    
    if split_pop: 
        r = jr.uniform(k2)
        pi_i = r * types.pi[i]
        pi_k = (1-r) * types.pi[i]
        pi = types.pi.at[k].set(pi_k)
        pi = pi.at[i].set(pi_i)
        types = eqx.tree_at(lambda types: types.pi, types, pi)
    
    return model, jnn.one_hot(i, num_classes=types.psi.shape[0])

def add_random_type(model, key):
    active = model.types.active
    inactive = 1.0 - active
    k = jr.choice(key, jnp.arange(active.shape[0]), p=inactive/inactive.sum())
    active = active.at[k].set(True)
    order = jnp.argsort(active, descending=True)
    types = model.types.replace(active=active)
    types = jax.tree.map(lambda x: x[order], types)
    return eqx.tree_at(lambda mdl: mdl.types, model, types)

def remove_type(model, key):
    active = model.types.active
    k = jr.choice(key, jnp.arange(active.shape[0]), p=active/active.sum())
    active = active.at[k].set(False)
    order = jnp.argsort(active, descending=True)
    types = model.types.replace(active=active)
    types = jax.tree.map(lambda x: x[order], types)
    return eqx.tree_at(lambda t: t.types, model, types)


min_prms = lambda prms_like: eqx.tree_at(
    lambda tree: [
        tree.types.theta, 
        tree.types.pi,
        tree.types.gamma
    ],
    jax.tree.map(lambda x: -jnp.inf, prms_like),
    [
        0.0,
        0.0,
        1e-4
    ]
)

max_prms = lambda prms_like: eqx.tree_at(
    lambda x: x.types.theta,
    jax.tree.map(lambda x:jnp.full_like(x, jnp.inf), prms_like),
    jnp.ones_like(prms_like.types.theta)
)

mask_prms = lambda prms_like: eqx.tree_at(
    lambda tree: tree.types.active,
    jax.tree.map(lambda x: True, prms_like),
    False
)

prms_sample_min = lambda prms_like: eqx.tree_at(
    lambda tree: [
        tree.types.theta, 
        tree.types.pi,
        tree.types.gamma
    ],
    jax.tree.map(lambda x: -1.0, prms_like),
    [
        0.0,
        0.0,
        1e-4
    ]
)

prms_sample_max = lambda prms_like: eqx.tree_at(
    lambda x: x.types.theta,
    jax.tree.map(lambda x:1.0, prms_like),
    0.0
)

def make_mutation_fn(prms_like: PyTree,
                     p_duplicate_split: float, 
                     p_duplicate_no_split: float,
                     p_mut: float,
                     p_rm: float,
                     p_add: float,
                     sigma_mut: float, 
                     prms_are_shaped: bool=True):

    flat_prms_like, shaper = ravel_pytree(prms_like)
    # ---
    clip_min = min_prms(prms_like)
    clip_max = max_prms(prms_like)
    mutation_mask = mask_prms(prms_like)
    sample_min, _ = ravel_pytree(jax.tree.map(
        lambda x, v: jnp.full_like(x, v),
        prms_like, prms_sample_min(prms_like)
    ))
    sample_max, _ = ravel_pytree(jax.tree.map(
        lambda x, v: jnp.full_like(x, v),
        prms_like, prms_sample_max(prms_like)
    ))

    # ---
    
    def _duplicate_split(prms, key):
        prms, _ = duplicate_type(prms, key, split_pop=True)
        return prms

    def _duplicate_no_split(prms, key):
        prms, _ = duplicate_type(prms, key, split_pop=False)
        return prms

    def _rm(prms, key):
        return remove_type(prms, key)

    def _add(prms, key):
        return add_random_type(prms, key)
        
    def _point_mut(prms, key):
        k1, k2 = jr.split(key)
        muts = jr.uniform(k1, flat_prms_like.shape, minval=sample_min, maxval=sample_max)
        locs = jr.bernoulli(k2, p=p_mut, shape=muts.shape).astype(float)
        muts = shaper(muts)
        locs = shaper(locs)
        mut_prms = jax.tree.map(
            lambda x, _x, m: jnp.where(m.astype(bool), _x, x),
            prms, muts, locs
        )
        return mut_prms

    def _mutate(prms, key)->jax.Array:
        # small perturb
        epsilon = jr.normal(key, flat_prms_like.shape) * sigma_mut
        epsilon = shaper(epsilon)
        prms = jax.tree.map(
            lambda x, eps, m, cmin, cmax: jnp.clip(x+eps,cmin,cmax) if m else x,  
            prms, epsilon, mutation_mask, clip_min, clip_max    
        )
        return prms

    #-------------------------------------------------------------------


    def _mutation_fn(prms: PyTree, key: jax.Array)->PyTree:

        prms = shaper(prms) if not prms_are_shaped else prms

        key, k1, k2 = jr.split(key, 3)

        if p_duplicate_split > 0.0:
            prms = jax.lax.cond(
                jr.uniform(k1)<p_duplicate_split,
                lambda prms, key: _duplicate_split(prms, key),
                lambda prms, key: prms,
                prms, k2
            )

        key, k1, k2 = jr.split(key, 3)

        if p_duplicate_no_split > 0.0:
            prms = jax.lax.cond(
                jr.uniform(k1)<p_duplicate_no_split,
                lambda prms, key: _duplicate_no_split(prms, key),
                lambda prms, key: prms,
                prms, k2
            )

        key, k1, k2 = jr.split(key, 3)

        if p_rm > 0.0:
            prms = jax.lax.cond(
                jr.uniform(k1)<p_rm,
                lambda prms, key: _rm(prms, key),
                lambda prms, key: prms,
                prms, k2
            )

        key, k1, k2 = jr.split(key, 3)

        if p_add > 0.0:
            prms = jax.lax.cond(
                jr.uniform(k1)<p_add,
                lambda prms, key: _add(prms, key),
                lambda prms, key: prms,
                prms, k2
            )

        k1, k2 = jr.split(key)
        if p_mut > 0.0:
            prms = _point_mut(prms, k1) #type:ignore
        if sigma_mut > 0.0:
            prms = _mutate(prms, k2)
        
        return prms

    #-------------------------------------------------------------------

    return _mutation_fn


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    mdl = Model_E(8, 2, key=jr.key(1))
    prms = eqx.filter(mdl, eqx.is_array)
    mutation_fn = make_mutation_fn(prms, 0.001, .001, .001, .001, 0.01, True)
    prms = mutation_fn(prms, jr.key(1))
    

