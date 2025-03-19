import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import equinox as eqx
import evosax as ex
from jaxtyping import PyTree
from typing import Callable
from flax import struct

@struct.dataclass
class GAState:
    mean: jax.Array
    archive: jax.Array
    fitness: jax.Array
    best_member: jax.Array
    best_fitness: jax.Array
    sigma: float
    gen_counter: int=0

@struct.dataclass
class GAParams:
    clip_min=-jnp.inf
    clip_max=jnp.inf

class GA(ex.Strategy):
    # ---
    def __init__(self, mutation_fn: Callable, init_prms: PyTree, popsize: int, elite_ratio: float=0.5, p_duplicate: float=0.01, 
                 sigma_init: float=0.01, sigma_decay: float=1., sigma_limit: float=0.01, prms_shaper=None):
        self.sigma_init = sigma_init
        self.sigma_decay = sigma_decay
        self.sigma_limit = sigma_limit
        self.p_duplicate = p_duplicate
        if prms_shaper is None:
            self.prms_shaper = ex.ParameterReshaper(init_prms, verbose=True)
        else:
            self.prms_shaper = prms_shaper
        super().__init__(popsize, num_dims=self.prms_shaper.total_params)
        self.init_prms = self.prms_shaper.flatten_single(init_prms)
        self.elite_popsize = int(popsize*elite_ratio)
        # --- mutation fn ---
        self.mutation_fn = mutation_fn
    # ---
    @property
    def params_strategy(self):
        return None
    # ---
    def ask(self, rng, state, params=None): #type:ignore
        if params is None:
            params = self.default_params

        # Generate proposal based on strategy-specific ask method
        x, state = self.ask_strategy(rng, state, params)

        return x, state
    # ---
    def ask_strategy(self, key, state, prms=None): #type:ignore
        kselect, kmut, kcross = jr.split(key, 3)
        prms = state.archive
        selected = jr.choice(kselect, prms, (self.popsize,))
        offsprings = jax.vmap(self.mutation_fn, in_axes=(0,0,None))(selected, jr.split(kmut, self.popsize), state)
        
        return offsprings, state
    # ---
    def tell_strategy(self, x, fitness, state, prms): #type:ignore
        fitness = jnp.concatenate([fitness, state.fitness])
        solution = jnp.concatenate([x, state.archive])
        # Select top elite from total archive info
        idx = jnp.argsort(fitness)[0 : self.elite_popsize]
        fitness = fitness[idx]
        archive = solution[idx]
        # Update mutation epsilon - multiplicative decay
        sigma = jnp.clip(state.sigma * self.sigma_decay, self.sigma_limit, jnp.inf)
        # Set mean to best member seen so far
        improved = fitness[0] < state.best_fitness
        best_mean = jax.lax.select(improved, archive[0], state.best_member)
        return state.replace(
            fitness=fitness, archive=archive, sigma=sigma, mean=best_mean
        )
    # ---
    def initialize_strategy(self, key, prms=None): #type:ignore
        prms = jnp.stack([self.init_prms]*self.elite_popsize)
        prms = self.prms_shaper.flatten(prms)
        fitness = jnp.full((self.elite_popsize,), jnp.inf)
        archive = prms
        return GAState(
            mean=self.init_prms, archive=archive, fitness=fitness, best_member=self.init_prms, #type:ignore
            best_fitness=jnp.inf, sigma=self.sigma_init #type:ignore
        )
    # ---