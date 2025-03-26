from jax.flatten_util import ravel_pytree
from ..eco.gridworld import Observation
from .ctrnn import CTRNN, CTRNNPolicy, CTRNNPolicyConfig

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import equinox as eqx
import equinox.nn as nn
import evosax as ex

from typing import Callable, NamedTuple
from jaxtyping import Float, Int, PyTree


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
    migration_field=migration_field, T_interactions=0.0, shape: str="square", *, key):
    
    N, _ = xs.shape
    T = temperature_decay ** t
    k = jnn.sigmoid((t - T_interactions)*10.0)
    mask = mask * jnn.sigmoid((t-thetas)*10.0)

    M = migration_field
    def M_(x):
        """Modified molecular field"""
        d = jnp.sum(jnp.square(x[None]-xs), axis=-1, keepdims=True) #N,1
        return M(x) + k*jnp.sum(zetas * jnp.exp(-d/(gammas+1e-6)) * mask[:,None], axis=0)

    def E(x, psi):
        """energy field"""
        return jnp.dot(M_(x), psi)

    dx = - jax.vmap(jax.grad(E))(xs, psis)
    dx_norm = jnp.linalg.norm(dx, axis=-1, keepdims=True)
    dx = jnp.where(dx_norm>0, dx/dx_norm, dx)
    dx = dx * mask[:,None]

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
    def __init__(self, dims, key):
        self.O = jnp.zeros((dims,dims))
    def __call__(self, x_pre, x_post):
        return x_pre @ self.O @ x_post

class MLPConn(nn.MLP):
    def __init__(self, dims, key):
        super().__init__(dims*2, "scalar", 16, 1, key=key)
    def __call__(self, x_pre, x_post): #type:ignore
        return super().__call__(jnp.concatenate([x_pre,x_post]))


dummy_policy_config = CTRNNPolicyConfig(lambda x:x, lambda x:x)
    
class Model_E(CTRNNPolicy):
    # --- params ---
    types: NeuronType
    connection_model: PyTree
    A: nn.MLP
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
    def __init__(self, n_types: int, n_synaptic_markers: int, max_nodes: int=32, sensory_dimensions: int=1, 
                 motor_dimensions: int=1, dt: float=0.1, dvpt_time: float=10., temperature_decay: float=1., extra_migration_fields: int=3,
                 N_gain: float=10.0, policy_cfg: CTRNNPolicyConfig=dummy_policy_config, body_shape: str="square", connection_model: str="xoxt", *, key: jax.Array):

        super().__init__(policy_cfg)
        
        k1, k2 = jr.split(key)

        n_fields = N_MORPHOGENS + extra_migration_fields
        self.migration_field = lambda x: jnp.concatenate([migration_field(x),jnp.zeros(extra_migration_fields)])
        
        types = NeuronType(
            pi = jnp.zeros(n_types),
            psi = jnp.zeros((n_types, n_fields)),
            gamma = jnp.zeros((n_types, n_fields))+0.001,
            zeta = jnp.zeros((n_types, n_fields)),
            omega = jnp.zeros((n_types, n_synaptic_markers)),
            theta = jnp.ones(n_types),
            active = jnp.zeros(n_types).at[0].set(1.),
            id_ = jnp.arange(n_types),
            s = jnp.zeros((n_types,sensory_dimensions)),
            m = jnp.zeros((n_types, motor_dimensions)),
            tau = jnp.ones(n_types),
            bias= jnp.zeros(n_types),
            gain = jnp.ones(n_types)
        )
        
        self.types = types
        if connection_model=="xoxt":
            self.connection_model = XOXT(n_synaptic_markers, k1)
        elif connection_model=="mlp":
            self.connection_model = MLPConn(n_synaptic_markers, k1)
        else:
            raise ValueError("no such conn model")

        self.A = nn.MLP(n_synaptic_markers+n_fields, n_synaptic_markers, 16, 1, key=k2)
        
        self.n_types = n_types
        self.max_nodes = max_nodes
        self.dt = dt
        self.dvpt_time = dvpt_time
        self.temperature_decay = temperature_decay
        self.N_gain = N_gain
        self.body_shape = body_shape
    # ---
    def initialize(self, key: jax.Array)->CTRNN:
        
        # 1. Initialize neurons
        x0 = jr.normal(key, (self.max_nodes, 2)) * 0.01
        node_type_ids = jnp.zeros(self.max_nodes)
        n_tot = 0
        pi = self.types.pi * self.types.active
        ns = jnp.round(pi * self.N_gain)
        for _, (n, msk) in enumerate(zip(ns, self.types.active)):
            node_type_ids = jnp.where(jnp.arange(self.max_nodes)<n_tot+n*msk, node_type_ids+1, node_type_ids)
            n_tot += n*msk
        node_type_ids = self.n_types - node_type_ids
        node_type_ids = jnp.where(node_type_ids < self.n_types, node_type_ids, -1).astype(int)
        node_types = jax.tree.map(lambda x: x[node_type_ids], self.types)

        
        # 2. Migrate
        step_fn = lambda i, x: migration_step(
            xs=x, t=self.dt*i, psis=node_types.psi, gammas=node_types.gamma, zetas=node_types.zeta, mask=node_types.active, 
            thetas=node_types.theta, dt=self.dt, temperature_decay=self.temperature_decay, migration_field=self.migration_field, 
            shape=self.body_shape, key=jr.key(1)
        )
        xs = jax.lax.fori_loop(0, int(self.dvpt_time//self.dt), step_fn, x0)
        
        # 3. Connect
        def molecular_field(x):
            """Modified molecular field"""
            d = jnp.sum(jnp.square(x[None]-xs), axis=-1, keepdims=True) #N,1
            return self.migration_field(x) + jnp.sum(node_types.zeta * jnp.exp(-d/(node_types.gamma+1e-6)) * node_types.active[:,None], axis=0)
        
        M = jax.vmap(molecular_field)(xs)
        g = jax.vmap(self.A)(jnp.concatenate([node_types.omega,M], axis=-1))
        W = jax.vmap(jax.vmap(self.connection_model, in_axes=(0,None)), in_axes=(None,0))(g,g)
        W = W * (node_types.active[:,None] * node_types.active[None])
        network = CTRNN(
            v=jnp.zeros(xs.shape[0]), 
            x=xs, W=W, tau=node_types.tau, gain=node_types.gain, bias=node_types.bias, 
            s=node_types.s, m=node_types.m, id_=node_types.id_, mask=node_types.active
        )
        
        return network
    # ---
    def partition(self):
        return eqx.partition(self, eqx.is_array)


# =================== HANDCRAFTED NETWORKS ==========================


def make_two_types(mdl, n_sensory_neurons, n_motor_neurons):
    n_synaptic_markers = mdl.types.omega.shape[-1]
    n_total = n_sensory_neurons + n_motor_neurons
    sensory_type = NeuronType(
        pi = n_sensory_neurons/mdl.N_gain, 
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
        pi = n_motor_neurons/mdl.N_gain,
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
    mdl = eqx.tree_at(lambda x: [x.types], mdl, [types])
    return mdl

def make_single_type(mdl, n_neurons):
    n_synaptic_markers = mdl.types.omega.shape[-1]
    sensorimotor_type = NeuronType(
        pi = n_neurons/mdl.N_gain, 
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
    mdl = eqx.tree_at(lambda x: [x.types], mdl, [types])
    return mdl

# ========================= INTERFACE =============================

def gridworld_sensory_interface(obs: Observation, ctrnn: CTRNN, fov: int=1):
    # ---
    assert ctrnn.s is not None
    # ---
    x = ctrnn.x
    # ---
    C = obs.chemicals # mC,W,W
    W = obs.walls
    mC, *_ = C.shape
    j, i = jnp.round((x*jnp.array([[1,-1]]))*fov+1).astype(int).T

    Ic = jnp.sum(C[:,i,j].T * ctrnn.s[:,:mC], axis=1) # chemical input
    Iw = W[i,j] * ctrnn.s[:,mC]
    Ii = jnp.sum(ctrnn.s[:, mC+1:] * obs.internal, axis=1) # internal input

    return Ic + Iw + Ii


def gridworld_motor_interface(ctrnn: CTRNN, threshold_to_move: float=1.0):
    # ---
    assert ctrnn.m is not None
    # --- 
    x = ctrnn.x
    m = ctrnn.m
    v = ctrnn.v
    # ---
    mask = jnp.max(jnp.abs(x),axis=-1)>0.5
    maximum_component = jnn.one_hot(jnp.argmax(jnp.abs(x), axis=-1), 2)
    x = x*maximum_component
    effects = jnp.round(x * jnp.array([[-1,1]]))[:,::-1] * mask[:,None]
    effects = effects * v[:,None] * m[:,None]
    move = jnp.sum(effects, axis=0)
    print(move)
    move = move * (jnp.abs(move).max()>=threshold_to_move)
    move_rep =  jnp.array([[1,0],[0,1],[-1,0],[0,-1],[0,0]])
    move_idx = jnp.argmin(jnp.square(move_rep - move[None]).sum(-1))
    return move_idx

# ========================= EVOLUTION =============================

def duplicate_type(model, key, split_pop=True):
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
    
    types = eqx.tree_at(lambda types: types.id_, types, jnp.arange(n_types))
    model = eqx.tree_at(lambda m: m.types, model, types)
    
    return model, jnn.one_hot(i, num_classes=types.psi.shape[0])

def add_random_type(model, key):
    active = model.types.active
    inactive = 1.0 - active
    k = jr.choice(key, jnp.arange(active.shape[0]), p=inactive/inactive.sum())
    active = active.at[k].set(1.0)
    order = jnp.argsort(active, descending=True)
    types = model.types._replace(active=active)
    types = jax.tree.map(lambda x: x[order], types)
    return eqx.tree_at(lambda mdl: mdl.types, model, types)

def remove_type(model, key):
    active = model.types.active
    k = jr.choice(key, jnp.arange(active.shape[0]), p=active/active.sum())
    active = active.at[k].set(0.0)
    order = jnp.argsort(active, descending=True)
    types = model.types._replace(active=active)
    types = jax.tree.map(lambda x: x[order], types)
    return eqx.tree_at(lambda t: t.types, model, types)

min_prms = lambda prms_like: eqx.tree_at(
    lambda tree: [
        tree.types.theta, 
        tree.types.pi,
        tree.types.gamma
    ],
    jax.tree.map(lambda x: jnp.full_like(x, -jnp.inf), prms_like),
    [
        jnp.zeros_like(prms_like.types.theta),
        jnp.zeros_like(prms_like.types.pi),
        jnp.full_like(prms_like.types.gamma, 1e-8)
    ]
)

max_prms = lambda prms_like: jax.tree.map(lambda x:jnp.full_like(x, jnp.inf), prms_like)

mask_prms = lambda prms_like: eqx.tree_at(
    lambda tree: [tree.types.id_, tree.types.active],
    jax.tree_map(lambda x: jnp.ones_like(x), prms_like),
    [jnp.zeros_like(prms_like.types.id_), jnp.zeros_like(prms_like.types.active)]
)

def mutate(prms: jax.Array,
           key: jax.Array, 
           p_duplicate: float, 
           p_mut: float,
           p_rm: float,
           p_add: float,
           sigma_mut: float, 
           shaper: ex.ParameterReshaper, 
           mutation_mask: jax.Array|None=None, 
           clip_min: jax.Array|None=None, 
           clip_max: jax.Array|None=None,
           split_pop_duplicate: bool=True):

    # ---
    prms_shaped = shaper.reshape_single(prms)
    if clip_min is None:
        clip_min, _ = ravel_pytree(min_prms(prms_shaped))
    if clip_max is None:
        clip_max, _ = ravel_pytree(max_prms(prms_shaped))
    if mutation_mask is None:
        mutation_mask, _ = ravel_pytree(mask_prms(prms_shaped))
    # ---
    
    def _duplicate(prms, key):
        prms, _ = duplicate_type(prms, key, split_pop=split_pop_duplicate)
        return prms

    def _rm(prms, key):
        return remove_type(prms, key)

    def _add(prms, key):
        return add_random_type(prms, key)
        

    def _mutate(prms, key):
        k1, k2 = jr.split(key)
        mut_locs = jr.bernoulli(k1, p_mut, prms.shape).astype(float)
        epsilon = jr.normal(k2, prms.shape) * sigma_mut * mutation_mask * mut_locs 
        prms = prms + epsilon
        prms = jnp.clip(prms, clip_min, clip_max)
        return prms


    key, k1, k2 = jr.split(key, 3)

    prms_shaped = jax.lax.cond(
        jr.uniform(k1)<p_duplicate,
        lambda prms, key: _duplicate(prms, key),
        lambda prms, key: prms,
        prms_shaped, k2
    )

    key, k1, k2 = jr.split(key, 3)

    prms_shaped = jax.lax.cond(
        jr.uniform(k1)<p_rm,
        lambda prms, key: _rm(prms, key),
        lambda prms, key: prms,
        prms_shaped, k2
    )

    key, k1, k2 = jr.split(key, 3)

    prms_shaped = jax.lax.cond(
        jr.uniform(k1)<p_add,
        lambda prms, key: _add(prms, key),
        lambda prms, key: prms,
        prms_shaped, k2
    )


    prms = shaper.flatten_single(prms_shaped) #type:ignore
    prms = _mutate(prms, key)
    
    return prms


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    x=jnp.mgrid[-10:11,-10:11].transpose((1,2,0)).reshape((-1,2)) / 10
    N=x.shape[0]
    ctrnn = CTRNN(
        x=x,
        v=jnp.zeros(N), 
        tau=None, #type:ignore
        gain=None, #type:ignore
        bias=None,#type:ignore
        W=None, #type:ignore
        m=jnp.ones(N), 
        s=jnp.ones((N,3)), 
        id_=None)
    
    obs = Observation(
        chemicals = jnp.array([
            [[0.,0.,0.],
             [0.,0.,0.],
             [0.,1.,0.]]
        ]),
        walls = jnp.array([[0.,0.,0.],
                           [0.,0.,0.],
                           [0.,0.,0.]]),
        internal=jnp.zeros(2)
    )
    I = gridworld_sensory_interface(obs, ctrnn)
    #plt.scatter(*ctrnn.x.T, c=I)
    #plt.show()

    ctrnn = ctrnn._replace(v=I)

    a = gridworld_motor_interface(ctrnn)
    print(a)
