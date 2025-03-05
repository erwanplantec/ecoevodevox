import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import equinox as eqx
import equinox.nn as nn


class ConvLSTMAgent(eqx.Module):
	# ---
	def __init__(self):
		pass