import wandb
import jax
import jax.numpy as jnp
from jaxtyping import PyTree
from typing import Tuple, TypeAlias, Callable, Optional
import equinox as eqx
import os
from jax.experimental import io_callback
import numpy as np
import pickle

from ..eco.gridworld import EnvState


class Logger:
	#-------------------------------------------------------------------
	def __init__(
		self, 
		metrics_fn: Callable, 
		ckpt_dir: str|None=None, 
		host_log_transform: Callable=lambda data: data):
		
		self.metrics_fn = metrics_fn
		self.ckpt_dir = ckpt_dir

		def _host_log_clbck(data):
			data = host_log_transform(data)
			wandb.log(data)
			return jnp.zeros((),dtype=bool)

		self.host_log_clbck = _host_log_clbck

		if ckpt_dir:
			def _host_ckpt_clbck(state):
				time = state.time
				filename = f"{ckpt_dir}/{int(time)}.pickle"
				with open(filename, "wb") as file:
					pickle.dump(state, file)
				return jnp.zeros((),dtype=bool)

			self.host_ckpt_clbck = _host_ckpt_clbck

	#-------------------------------------------------------------------
	
	def log(self, state, step_data):

		data = self.metrics_fn(state, step_data)
		_ = io_callback(self.host_log_clbck, jax.ShapeDtypeStruct((),bool), data)

	#-------------------------------------------------------------------

	def ckpt(self, state: EnvState, **other):

		data = {"env_state": state, **other}
		_ = io_callback(self.host_ckpt_clbck, jax.ShapeDtypeStruct((),bool), data)
			



