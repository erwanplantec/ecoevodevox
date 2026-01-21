from flax.struct import PyTreeNode
from jaxtyping import PyTree
import jax

from ..core import Observation, PolicyState, SensoryState, PolicyInput, Float16

class SensoryInterface(PyTreeNode):
	#-------------------------------------------------------------------
	def encode(self, obs: Observation, policy_state: PolicyState, sensory_state: SensoryState)->tuple[PolicyInput,Float16,SensoryState,dict]:
		"""
		maps environment observation to policy input
		"""
		raise NotImplementedError
	#-------------------------------------------------------------------
	def init(self, policy_state: PolicyState, key: jax.Array)->SensoryState:
		return None
	#-------------------------------------------------------------------