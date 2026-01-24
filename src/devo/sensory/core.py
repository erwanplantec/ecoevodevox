from flax.struct import PyTreeNode
from jaxtyping import PyTree
import jax

from ..core import Observation, NeuralState, SensoryState, NeuralInput, Float16

class SensoryInterface(PyTreeNode):
	#-------------------------------------------------------------------
	def encode(self, obs: Observation, neural_state: NeuralState, sensory_state: SensoryState)->tuple[NeuralInput,Float16,SensoryState,dict]:
		"""
		maps environment observation to policy input
		"""
		raise NotImplementedError
	#-------------------------------------------------------------------
	def init(self, neural_state: NeuralState, key: jax.Array)->SensoryState:
		return None
	#-------------------------------------------------------------------