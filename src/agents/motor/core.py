from typing import Any
from flax.struct import PyTreeNode
from jaxtyping import PyTree, Float
import jax

from ..core import Action, PolicyState, MotorState, Info, Body

class MotorInterface(PyTreeNode):
	#-------------------------------------------------------------------
	def decode(self, policy_state: PolicyState, motor_state: MotorState)->tuple[Action,Float,MotorState,Info]:
		"""decodes policy state into action"""
		raise NotImplementedError
	#-------------------------------------------------------------------
	def move(self, action: Action, body: Body)->Body:
		"""computes effect of action"""
		raise NotImplementedError
	#-------------------------------------------------------------------
	def init(self, policy_state: PolicyState, key: jax.Array)->MotorState: 
		return None
	#-------------------------------------------------------------------