from .base import DevelopmentalModel
from .rand import RAND
from .direct import DirectRNN, DirectCTRNN

import jax


class DummyEncodingModel(DevelopmentalModel):
	#-------------------------------------------------------------------
	def __init__(self, *args, **kwargs):
		pass
	#-------------------------------------------------------------------
	def __call__(self, key: jax.Array|None=None):
		return None

encoding_models = {
	"rand": RAND,
	"direct_rnn": DirectRNN,
	"direct_ctrnn": DirectCTRNN,
	"dummy": DummyEncodingModel,
}