from .base import NN, SENN, Policy
from .rnn import RNN, SERNN, RNNPolicy
from .ctrnn import CTRNNPolicy, CTRNN, SECTRNN

import equinox as eqx
from jax.flatten_util import ravel_pytree

nn_models = {
	"rnn": RNNPolicy,
	"ctrnn": CTRNNPolicy,
}

def make_apply_init(model, apply_method: str="__call__", init_method: str="init", reshape_prms: bool=False):
	"""Create init and apply functions from equinox style model 
	(corresponding methods are model.initialize and mode.__call__)"""
	if hasattr(model, "partition"):
		prms, sttcs = model.partition()
	else:
		prms, sttcs = eqx.partition(model, eqx.is_array)

	_, shaper = ravel_pytree(prms)

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