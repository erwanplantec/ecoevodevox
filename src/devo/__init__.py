from .base import DevelopmentalModel
from .rand import RAND
from .hypernetwork import HyperRNN
from .direct import DirectRNN, DirectCTRNN

import jax


class DummyEncodingModel(DevelopmentalModel):
	#-------------------------------------------------------------------
	def __init__(self, *args, **kwargs):
		pass
	#-------------------------------------------------------------------
	def __call__(self, key: jax.Array|None=None): #type:ignore
		return None

encoding_models = {
	"rand": RAND,
	"hyper_rnn": HyperRNN,
	"direct_rnn": DirectRNN,
	"direct_ctrnn": DirectCTRNN,
	"dummy": DummyEncodingModel,
}