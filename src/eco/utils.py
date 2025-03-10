import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn


def minivmap(func, minibatch_size, check_func=None, default_output=None, func_takes_batch=False):
	"""
	Args
		func (callable)
		minibatch_size (int)
		check_func (callable) : Don't execute batched_func
		default_otput (any) : output of func of check_func(args)==False
		func_takes_natch (bool)- whether func can admit batch dimension already. If False, func will be vmapped
	"""

	if not func_takes_batch:
		batched_func = jax.vmap(func)
	else:
		batched_func = func

	default_output = jax.tree.map(lambda x: jnp.stack([x for _ in range(minibatch_size)]), default_output)

	if check_func is not None:
		_batched_func = batched_func
		def batched_func(*args):
			return jax.lax.cond(
				check_func(*args), 
				_batched_func,
				lambda *_: default_output,
				*args
			)


	def mapped_func(*args):
		args_minibatches = jax.tree.map(lambda x: x.reshape((-1, minibatch_size, *x.shape[1:])), args) # n, mb, ...
		_, results = jax.lax.scan(
			lambda _, args: (None, batched_func(*args)),
			None,
			args_minibatches
		)
		results = jax.tree.map(lambda x: x.reshape((-1, *x.shape[2:])), results)
		return results

	return mapped_func



import wandb

class EnvLogWrapper:
	# ---
	def __init__(self, env):
		self.env = env
	# ---
	def step():
		pass
