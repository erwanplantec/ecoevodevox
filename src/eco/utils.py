import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import jax.scipy as jsp
import equinox.nn as nn
from functools import partial


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


f16, f32, i8, i16, i32, i64 = jnp.float16, jnp.float32, jnp.uint8, jnp.uint16, jnp.uint32, jnp.uint64
MAX_INT16 = jnp.iinfo(jnp.uint16).max
boolean_maxpool = lambda x: nn.Pool(init=False, operation=jnp.logical_or, num_spatial_dims=2, padding=1, kernel_size=3)(x[None])[0]
convolve = partial(jsp.signal.convolve, mode="same")

def neighbor_states_fn(x, include_center=True, neighborhood="moore"):
	if x.ndim>2:
		extra_offs = (0,)*(x.ndim-2)
	else:
		extra_offs = ()
	if neighborhood=="moore":
		shifts = [(*extra_offs, 0,1), (*extra_offs, 1,0), (*extra_offs, 0,-1), (*extra_offs, -1,0)]
	elif neighborhood=="vn":
		shifts = [(*extra_offs, di, dj) for di in [-1,0,1] for dj in [-1,0,1]]
	else:
		raise ValueError(f"neighborhood {neighborhood} is not valid. must be either 'moore' or 'vn'")
	output = jnp.stack([jnp.roll(x, shift) for shift in shifts])
	if include_center:
		output = jnp.concatenate([x[None], output], axis=0)
	return output

def moore_neighborhood(x, i, j):
	C, H, W = x.shape
	return jax.lax.dynamic_slice(x, [jnp.array(0,dtype=i16),i,j], [C,3,3])

def k_neighborhood(x, i, j, k=1):
	"""return size k neighborhood (moore=>k=1)"""
	C, H, W = x.shape
	window_sz = k*2+1
	return jax.lax.dynamic_slice(x, [jnp.array(0,dtype=i16),i,j], [C,window_sz,window_sz])