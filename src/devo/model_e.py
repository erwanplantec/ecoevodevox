from .policy import CTRNN, CTRNNPolicy, CTRNNPolicyConfig

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import equinox as eqx
import equinox.nn as nn
import evosax as ex

from typing import NamedTuple
from jaxtyping import Float, Int


class NeuronType(NamedTuple):
    # ---
    pi: Float
    active: Float
    id_: Int
    # -- Migration Parameters ---
    psi: Float # morphogens affinity
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

def morphogen_field(x):
    return jnp.array([x[0], x[1], x[0]*x[1], jnp.abs(x[0]), jnp.abs(x[1]), jnp.linalg.norm(x), (jnp.linalg.norm(x)-0.5)**2])

N_MORPHOGENS = 7

def safe_norm(x):
    return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)

def repulsion(x, xs, gamma, mask):
    dx = x[None] - xs #N,2
    d = jnp.linalg.norm(dx, axis=-1, keepdims=True) #N,1
    rep = 1 - jnn.sigmoid((d - gamma)*30.) #N,1
    dx_norm = dx / (jnp.linalg.norm(dx, axis=-1, keepdims=True)+1e-8)
    force = jnp.sum((dx_norm* rep) * mask[:,None], axis=0)
    return force

@jax.jit
def migration_step(x, t, psi, gamma, mask, theta=None, dt=0.1, alpha=1., beta=1., temperature_decay=1.0, *, key):
    
    N, _ = x.shape
    T = temperature_decay ** t

    # Morphogen gradient forces
    dx_m = jax.grad(lambda x, psi: (jax.vmap(morphogen_field)(x) * psi).sum())(x, psi)

    # Repulsion forces
    dx_r = jax.vmap(repulsion, in_axes=(0,None,0, None))(x,x,gamma,mask)

    # Speed
    speed = jnp.ones((N,1)) if theta is None else jnn.sigmoid((t - theta)*10.)[:,None]

    dx = alpha * dx_m + beta * dx_r 
    dx = safe_norm(dx) * speed
    dx = dx * mask[:,None] 
    
    x = x + dx*dt*T*speed
    norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
    x = jnp.where(norm>1., x / norm, x)
    
    return x


dummy_policy_config = CTRNNPolicyConfig(lambda x:x, lambda x:x)
    
class Model_E(CTRNNPolicy):
    # --- params ---
    types: NeuronType
    N: jax.Array
    O: jax.Array
    A: nn.MLP
    alpha: jax.Array
    beta: jax.Array
    # --- statics ---
    n_types: int
    max_nodes: int
    dt: float
    dvpt_time: float
    temperature_decay: float
    # ---
    def __init__(self, n_types: int, n_morphogens: int, n_synaptic_markers: int, max_nodes_per_type: int=32, 
                 alpha: float=1., beta: float=0.5, dt: float=0.1, dvpt_time: float=10., temperature_decay: float=1., 
                 policy_cfg: CTRNNPolicyConfig=dummy_policy_config, *, key: jax.Array):

        super().__init__(policy_cfg)
        
        k1, k2 = jr.split(key)
        
        types = NeuronType(
            pi = jnp.zeros(n_types),
            psi = jnp.zeros((n_types, n_morphogens)),
            omega = jnp.zeros((n_types, n_synaptic_markers)),
            gamma = jnp.zeros((n_types,)),
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
        self.A = nn.MLP(n_synaptic_markers+n_morphogens, n_synaptic_markers, 64, 1, key=k2)
        self.N = jnp.zeros(())
        
        self.n_types = n_types
        self.max_nodes = max_nodes_per_type * n_types
        self.alpha = jnp.ones(())*alpha
        self.beta = jnp.ones(())*beta
        self.dt = dt
        self.dvpt_time = dvpt_time
        self.temperature_decay = temperature_decay
    # ---
    def initialize(self, key: jax.Array)->CTRNN:
        
        # 1. Initialize neurons
        x0 = jr.normal(key, (self.max_nodes, 2)) * 0.01
        node_type_ids = jnp.zeros(self.max_nodes)
        n_tot = 0
        pi = self.types.pi / jnp.sum(self.types.pi * self.types.active)
        n = jnp.round(self.N * 10.0 * pi)
        for _, (n, msk) in enumerate(zip(n, self.types.active)):
            node_type_ids = jnp.where(jnp.arange(self.max_nodes)<n_tot+n*msk, node_type_ids+1, node_type_ids)
            n_tot += n*msk
        node_type_ids = self.n_types - node_type_ids
        node_type_ids = jnp.where(node_type_ids < self.n_types, node_type_ids, -1).astype(int)
        node_types = jax.tree.map(lambda x: x[node_type_ids], self.types)
        
        # 2. Migrate
        step_fn = lambda i, x: migration_step(
            x=x, t=self.dt*i, psi=node_types.psi, gamma=node_types.gamma, mask=node_types.active, 
            theta=node_types.theta, dt=self.dt, temperature_decay=self.temperature_decay, key=jr.key(1), 
            alpha=self.alpha, beta=self.beta
        )
        x = jax.lax.fori_loop(0, int(self.dvpt_time//self.dt), step_fn, x0)
        
        # 3. Connect
        M = jax.vmap(morphogen_field)(x)
        g = jax.vmap(self.A)(jnp.concatenate([node_types.omega,M], axis=-1))
        W = jnp.matmul(jnp.matmul(g, self.O), g.T)
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


def make_two_types(mdl, n_sensory_neurons, n_motor_neurons):
    n_synaptic_markers = mdl.types.omega.shape[-1]
    n_total = n_sensory_neurons + n_motor_neurons
    sensory_type = NeuronType(
        pi = n_sensory_neurons/n_total, 
        id_ = 0,
        psi = jnp.array([0.,1.,0., 0., 0., 0., 0.]),
        theta = 1.,
        omega = jnn.one_hot(0, n_synaptic_markers),
        gamma = 0.4,
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
        psi = jnp.array([0.,-1.,0., 0., 0., 0., 0.]),
        theta = 1.,
        omega = jnn.one_hot(1, n_synaptic_markers),
        gamma = 0.5,
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
        psi = jnp.zeros(N_MORPHOGENS),
        theta = 1.,
        omega = jnn.one_hot(0, n_synaptic_markers),
        gamma = 0.5,
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
    