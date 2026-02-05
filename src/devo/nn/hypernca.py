from .ctrnn import IndirectCTRNN, IndirectCTRNNState

import jax, jax.numpy as jnp, jax.random as jr, jax.nn as jnn
import equinox as eqx
import equinox.nn as nn
from typing import Callable
from jax.flatten_util import ravel_pytree

class PerceptionModel(eqx.Module):
    # ------------------------------------------------------------------
    convnet: nn.Conv2d
    activation_fn: Callable
    out_channels: int
    # ------------------------------------------------------------------
    def __init__(self, channels: int, perception_channels, mix_channels: bool=False, 
                 activation_fn: Callable|str="relu", *, key: jax.Array):

        out_channels = perception_channels if mix_channels else perception_channels*channels
        groups = 1 if mix_channels else channels
        self.convnet = nn.Conv2d(in_channels=channels, 
                                 out_channels=out_channels,
                                 kernel_size=3, 
                                 stride=1, 
                                 padding="SAME", 
                                 groups=groups, 
                                 key=key)

        self.activation_fn = getattr(jnn, activation_fn) if isinstance(activation_fn, str) else activation_fn
        self.out_channels = out_channels
    # ------------------------------------------------------------------
    def __call__(self, X: jax.Array):
        return self.activation_fn(self.convnet(X))

class UpdateModel(eqx.Module):
    # ------------------------------------------------------------------
    layers: list[nn.Conv2d]
    activation_fn: Callable
    final_activation_fn: Callable
    # ------------------------------------------------------------------
    def __init__(self, channels: int, perception_channels: int, nb_layers: int, 
                 activation_fn: Callable|str="tanh", final_activation_fn: Callable|str="identity",
                 *, key: jax.Array):
        keys = jr.split(key, nb_layers)
        sizes = [perception_channels] + [channels]*nb_layers
        self.layers = [
            nn.Conv2d(sizes[i], sizes[i+1], 1, 1,  key=keys[i])
            for i in range(nb_layers)
        ]
        self.activation_fn = getattr(jnn, activation_fn) if isinstance(activation_fn, str) else activation_fn
        self.final_activation_fn = getattr(jnn, final_activation_fn) if isinstance(final_activation_fn, str) else final_activation_fn
    # ------------------------------------------------------------------
    def __call__(self, X: jax.Array, P: jax.Array):
        y = P
        for layer in self.layers:
            y = self.activation_fn(layer(y))
        return X + self.final_activation_fn(y)



class NCA(eqx.Module):
    # ------------------------------------------------------------------
    update_model: UpdateModel
    perception_model: PerceptionModel
    # ------------------------------------------------------------------
    def __init__(self, channels: int, perception_channels: int, update_layers: int=2, 
                 update_kws: dict={}, perception_kws: dict={}, *, key: jax.Array):
        key_update, key_perception = jr.split(key)
        self.perception_model = PerceptionModel(channels, perception_channels, 
                                                 key=key_perception, **perception_kws)
        self.update_model = UpdateModel(channels, self.perception_model.out_channels, update_layers, 
                                        key=key_update, **update_kws)
    # ------------------------------------------------------------------
    def __call__(self, X: jax.Array):
        P = self.perception_model(X)
        X = self.update_model(X, P)
        return X

class NeuronNCAState(IndirectCTRNNState):
    x: jax.Array
    s: jax.Array

class NeuronNCA(IndirectCTRNN):
    # ------------------------------------------------------------------
    nca: NCA
    wiring_rules: jax.Array
    dev_steps: int
    size: int
    synapse_channels: int
    total_channels: int
    channel_partitioner: Callable
    # ------------------------------------------------------------------
    def __init__(self, size: int, synapse_channels: int, extra_channels: int, perception_channels: int,
                 update_layers: int, dev_steps: int, nb_wiring_rules, update_kws: dict={}, 
                 perception_kws: dict={}, ctrnn_dt: float=0.03, ctrnn_T: float=1.0, 
                 ctrnn_activation: Callable|str="tanh", *, key: jax.Array):
        super().__init__(ctrnn_dt, ctrnn_T, ctrnn_activation)

        channels = {"synapse": jnp.zeros(synapse_channels), 
                    "logtau": jnp.zeros(()),
                    "bias": jnp.zeros(()),
                    "sensory": jnp.zeros((1,)),
                    "motor": jnp.zeros(()),
                    "other": jnp.zeros((extra_channels,))}
        flattened_channels, self.channel_partitioner = ravel_pytree(channels)
        total_channels = len(flattened_channels)

        key_nca, key_rules = jr.split(key)
        self.nca = NCA(total_channels, perception_channels, update_layers, update_kws, perception_kws, key=key_nca)
        self.wiring_rules = jr.normal(key_rules, (nb_wiring_rules, synapse_channels, synapse_channels)) * 0.1
        # ---
        self.dev_steps = dev_steps
        self.size = size
        self.total_channels = total_channels
        self.synapse_channels = synapse_channels
        # ---
    # ------------------------------------------------------------------
    def init(self, key: jax.Array) -> NeuronNCAState:
        x_axis = jnp.linspace(0, 1, self.size)[None, None].repeat(self.size, 1)
        y_axis = jnp.linspace(0, 1, self.size)[None, :, None].repeat(self.size, 2)
        X_init = jnp.concatenate([
            x_axis,
            y_axis,
            jnp.abs(x_axis * 2. - 1.),
            jnp.abs(y_axis * 2. - 1.),
            jnp.zeros((self.total_channels-4, self.size, self.size))
        ], axis=0)

        def _step(i, X):
            X = self.nca(X)
            return X

        X_final = jax.lax.fori_loop(0, self.dev_steps, _step, X_init)
        channels = jax.vmap(self.channel_partitioner)(X_final.reshape(-1, self.size**2).T)
        def _compute_rule(X, O):
            return X @ O @ X.T
        W_rules = jax.vmap(_compute_rule, in_axes=(None,0))(channels["synapse"].reshape(self.synapse_channels, -1).T, self.wiring_rules)
        W = jnp.sum(W_rules, axis=0)
        bias = channels["bias"].reshape(-1)
        tau = jnp.exp(channels["logtau"]).reshape(-1) + 1e-5
        coords = jnp.linspace(-1.0, 1.0, self.size)
        xy = jnp.stack([coords[None].repeat(self.size, 0), coords[:,None].repeat(self.size, 1)], axis=2).reshape(-1, 2)
        s = jnn.sigmoid(channels["sensory"])

        return NeuronNCAState(v=jnp.zeros_like(tau), W=W, tau=tau, gain=jnp.ones_like(tau), bias=bias, mask=jnp.ones_like(tau), x=xy, s=s)
        


        

