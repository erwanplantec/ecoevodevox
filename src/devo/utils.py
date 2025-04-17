import jax
import jax.numpy as jnp
import jax.nn as jnn
import equinox as eqx
from jax.flatten_util import ravel_pytree

def make_apply_init(model, apply_method: str="__call__", init_method: str="initialize", reshape_prms: bool=True):
	"""Create init and apply functions from equinox style model 
	(corresponding methods are model.initialize and mode.__call__)"""
	if hasattr(model, "partition"):
		prms, sttcs = model.partition()
	else:
		prms, sttcs = eqx.partition(model, eqx.is_array)

	_, shaper = ravel_pytree(prms) #type:ignore

	def apply_fn(prms, *args, **kwargs):
		if reshape_prms: prms = shaper(prms)
		mdl = eqx.combine(prms, sttcs)
		f = getattr(mdl, apply_method)
		return f(*args, **kwargs)

	def init_fn(prms, *args, **kwargs):
		if reshape_prms: prms = shaper(prms)
		mdl = eqx.combine(prms, sttcs)
		f = getattr(mdl, init_method)
		return f(*args, **kwargs)

	return apply_fn, init_fn


def square_domain(p, i=0.5):
	"""
	return the one_hot_encoding of class of the domain in which x lies
	classes : [0: center, 1: upper, 2: right, 3: bottom, 4: left]
	"""
	x, y = p
	x_ = -x
	y_ = -y
	c = (x>-i) & (x<i) & (y>-i) & (y<i)
	u = (y>i) & (y>=x) & (x>-y)
	r = (x>i) & (x>=y) & (y>-x)
	b = (y_>i) & (y_>=x) & (x>=-y_)
	l = (x_>i) & (x_>=y) & (y>-x_)
	return jnp.stack([c,u,r,b,l])


if __name__ == '__main__':
	import matplotlib.pyplot as plt
	p = jnp.mgrid[-100:100,-100:100]/100
	print(p.shape)
	domain_classes = square_domain(p)
	fig, ax = plt.subplots(1, domain_classes.shape[0], figsize=(15,3))
	for i, msk in enumerate(domain_classes):
		ax[i].imshow(msk, origin="lower")
	plt.show()
