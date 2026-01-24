"""Summary
"""
from flax.struct import PyTreeNode
import jax, jax.numpy as jnp, jax.random as jr, jax.nn as jnn
import equinox as eqx
from typing import Callable
from jaxtyping import Float

from .core import NeuralModel


class CTRNNState(PyTreeNode):

    """Summary
    """
    
    v: jax.Array

class CTRNN(NeuralModel):

    """Summary
    
    Attributes:
        activation_fn (TYPE): Description
        bias (TYPE): Description
        dt (TYPE): Description
        gain (TYPE): Description
        iters (TYPE): Description
        nb_neurons (TYPE): Description
        tau (TYPE): Description
        W (TYPE): Description
    """
    
    # ------------------------------------------------------------------
    nb_neurons: int
    activation_fn: Callable
    dt: float
    W: jax.Array
    tau: jax.Array
    gain: jax.Array
    bias: jax.Array
    iters: int
    # ------------------------------------------------------------------
    def __init__(self, 
                 nb_neurons: int=64,
                 dt: float=0.1, 
                 T: float=1.0, 
                 activation_fn: str|Callable="tanh",
                 *,
                 key: jax.Array):
        """Summary
        
        Args:
            nb_neurons (int): Description
            dt (float, optional): Description
            T (float, optional): Description
            activation_fn (str | Callable, optional): Description
            key (jax.Array): Description
        """
        self.nb_neurons = nb_neurons
        self.activation_fn = getattr(jnn, activation_fn) if isinstance(activation_fn, str) else activation_fn
        self.dt = dt
        self.iters = int(T//dt)
        key_tau, key_bias, key_gain, key_W = jr.split(key, 4)
        self.tau = jr.uniform(key_tau, (nb_neurons,), minval=0.01, maxval=1.0)
        self.gain = jnp.ones(nb_neurons)
        self.bias = jnp.zeros(nb_neurons)
        W_init = jnn.initializers.orthogonal()
        self.W = W_init(key_W, (nb_neurons, nb_neurons))
    # ------------------------------------------------------------------
    def __call__(self, x: jax.Array, state: CTRNNState, key: jax.Array):
        """Summary
        
        Args:
            x (jax.Array): Description
            state (CTRNNState): Description
            key (jax.Array): Description
        
        Returns:
            TYPE: Description
        """
        # forward network
        forward_fn = lambda i, v: CTRNN.forward(state.v, x, self.W, self.bias, self.tau, self.gain, self.dt, self.activation_fn)
        v = jax.lax.fori_loop(0, self.iters, forward_fn, state.v)
        return state.replace(v=v), 0.0
    # ------------------------------------------------------------------
    def init(self, key: jax.Array) -> CTRNNState:
        """Summary
        
        Args:
            key (jax.Array): Description
        
        Returns:
            CTRNNState: Description
        """
        return CTRNNState(v=jnp.zeros(self.nb_neurons))
    # ------------------------------------------------------------------
    @classmethod
    def forward(cls, 
                v: jax.Array, 
                x: jax.Array, 
                W: jax.Array,
                bias: jax.Array,
                tau: jax.Array, 
                gain: jax.Array,  
                dt: float=0.1, 
                f: Callable=jnn.tanh,
                mask: jax.Array|None=None)->jax.Array:
        """ctrnn update step
        
        Args:
            v (jax.Array): activation vector
            x (jax.Array): input vector
            W (jax.Array): weight matrix
            bias (jax.Array): Description
            tau (jax.Array): Description
            gain (jax.Array): Description
            dt (float, optional): Description
            f (Callable, optional): Description
            mask (jax.Array | None, optional): Description
        
        Returns:
            jax.Array: Description
        """
        mask = mask if mask is not None else jnp.ones_like(v)
        y  = f(gain*(v+bias)) * mask
        dv = - (v) + jnp.matmul(W, y) + x
        v  = v + dt * (1/tau)  * dv
        v = jnp.clip(v, -5.0, 5.0)
        v = v * mask
        return v



class IndirectCTRNNState(CTRNNState):

    """Summary
    """
    
    v: jax.Array
    W: jax.Array
    tau: jax.Array
    gain: jax.Array
    bias: jax.Array
    mask: jax.Array|None

class IndirectCTRNN(NeuralModel):
    
    # ------------------------------------------------------------------
    activation_fn: Callable
    dt: float
    T: float
    # ------------------------------------------------------------------
    def __init__(self, dt: float, T: float, activation_fn: str|Callable):
        """Summary
        
        Args:
            dt (float): Description
            T (float): Description
            activation_fn (str | Callable): Description
        """
        self.dt = dt
        self.T = T
        self.activation_fn = getattr(jnn, activation_fn) if isinstance(activation_fn, str) else activation_fn
    # ------------------------------------------------------------------
    def __call__(self, x: jax.Array, state: IndirectCTRNNState, key: jax.Array) -> tuple[IndirectCTRNNState, Float]:
        """Summary
        
        Args:
            x (jax.Array): Description
            state (IndirectCTRNNState): Description
            key (jax.Array): Description
        
        Returns:
            tuple[IndirectCTRNNState, Float]: Description
        """
        forward = lambda _, v: CTRNN.forward(v, x, state.W, state.bias, state.tau, state.gain, self.dt, self.activation_fn, state.mask)
        v = jax.lax.fori_loop(0, int(self.T/self.dt), forward, state.v)
        return state.replace(v=v), 0.0
    # ------------------------------------------------------------------
    def init(self, key: jax.Array) -> IndirectCTRNNState:
        """Summary
        
        Args:
            key (jax.Array): Description
        
        Raises:
            NotImplementedError: Description
        """
        raise NotImplementedError























