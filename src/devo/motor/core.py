from typing import Any
from flax.struct import PyTreeNode
from jaxtyping import PyTree, Float
import jax

from ..core import Action, NeuralState, MotorState, Info, Body

class MotorInterface(PyTreeNode):
	#-------------------------------------------------------------------
	def actuate(self, neural_state: NeuralState, motor_state: MotorState, body: Body, key: jax.Array)->tuple[Body,Float,MotorState,Info]:
		"""Turn the neural + motor state into the *next body* and the action's energy cost.

		Fuses the old `decode` (state -> action) and `move` (action -> body) into one call that
		has the body — hence its size, for size-dependent kinematics/cost — and a `key`, for
		stochastic motors. Returns the RAW moved body (unwrapped); the toroidal wrap and wall
		handling stay world-level (see `Simulator.apply_agents_actions`). `info["action"]` carries
		the decoded action for callers doing behaviour analysis.
		"""
		raise NotImplementedError
	#-------------------------------------------------------------------
	def move(self, action: Action, body: Body)->Body:
		"""Pure kinematics: apply an action to a body. Used internally by `actuate`."""
		raise NotImplementedError
	#-------------------------------------------------------------------
	def init(self, policy_state: NeuralState, *, key: jax.Array)->MotorState: 
		return None
	#-------------------------------------------------------------------