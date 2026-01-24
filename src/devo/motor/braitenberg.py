from typing import Literal
from flax.struct import PyTreeNode
import jax
from jax import numpy as jnp, random as jr, nn as jnn
import equinox as eqx
import equinox.nn as nn
from flax import struct
from jaxtyping import Float
import numpy as np

from .core import MotorInterface, Action, Body, NeuralState, MotorState, Info

class BraitenbergSEMotorState(struct.PyTreeNode):
	on_right_motor: jax.Array
	on_left_motor: jax.Array

type BraitenbergMotorState = BraitenbergSEMotorState|None

class BraitenbergMotorInterface(MotorInterface):
	# ------------------------------------------------------------------
	dt: float=0.1
	interface: Literal["se", "direct"]="direct"
	wheel_speed_gain: float=0.1
	motor_energy_cost: float=0.1
	which_motorneurons: str="last"
	max_wheel_speed: float=1.0
	# --- se
	max_neuron_force: float=1.0
	motor_expression_threshold: float=0.1
	max_distance_to_motor: float=0.3
	# ------------------------------------------------------------------

	def init(self, neural_state: NeuralState, *, key: jax.Array) -> BraitenbergMotorState:

		if self.interface=="se":
			return self._init_se(neural_state, key=key)
		elif self.interface=="direct":
			return self._init_direct(neural_state, key=key)
		else:
			raise ValueError(f"interface {self.interface} is not valid for braitenberg motor interface")

	# ------------------------------------------------------------------

	def _init_se(self, neural_state: NeuralState, *, key: jax.Array) -> BraitenbergSEMotorState:

		# ---
		assert hasattr(neural_state, "x") #make sure network is spatially embedded
		assert hasattr(neural_state, "m") #make sure neurons have motor propoerties
		# ---
		xs = neural_state.x
		left_motor_pos = jnp.array([-1.0, 0.0])
		right_motor_pos = jnp.array([1.0, 0.0])

		dist_to_left_motor = jnp.linalg.norm(xs-left_motor_pos[None], axis=-1)
		on_left_motor = dist_to_left_motor < self.max_distance_to_motor

		dist_to_right_motor = jnp.linalg.norm(xs-right_motor_pos[None], axis=-1)
		on_right_motor = dist_to_right_motor < self.max_distance_to_motor

		is_motor = neural_state.m > self.motor_expression_threshold

		return BraitenbergSEMotorState(on_right_motor&is_motor, on_left_motor&is_motor)

	# ------------------------------------------------------------------

	def _init_direct(self, neural_state: NeuralState, *, key: jax.Array) -> BraitenbergMotorState:

		return None

	# ------------------------------------------------------------------

	def decode(self, neural_state: NeuralState, motor_state: MotorState) -> tuple[Action, Float, MotorState, Info]:
		
		if self.interface == "se":
			return self._decode_se(neural_state, motor_state)
		elif self.interface == "direct":
			return self._decode_direct(neural_state, motor_state)
		else:
			raise ValueError(f"interface {self.interface} is not valid for braitenberg motor interface")

	# ------------------------------------------------------------------

	def _decode_se(self, neural_state: NeuralState, motor_state: MotorState) -> tuple[Action, Float, MotorState, Info]:

		left_motor_activations = jnp.where(motor_state.on_left_motor, neural_state.v*neural_state.m, 0.0); assert isinstance(left_motor_activations, jax.Array)
		left_motor_activation = jnp.clip(left_motor_activations, -self.max_neuron_force, self.max_neuron_force).sum()

		right_motor_activations = jnp.where(motor_state.on_right_motor, neural_state.v*neural_state.m, 0.0); assert isinstance(right_motor_activations, jax.Array)
		right_motor_activation = jnp.clip(right_motor_activations, -self.max_neuron_force, self.max_neuron_force).sum()

		left_wheel_speed = jnp.clip(left_motor_activation*self.wheel_speed_gain, -self.max_wheel_speed, self.max_wheel_speed)
		right_wheel_speed = jnp.clip(right_motor_activation*self.wheel_speed_gain, -self.max_wheel_speed, self.max_wheel_speed)

		action = jnp.array([left_wheel_speed, right_wheel_speed], dtype=jnp.float16)
		action_norm = jnp.linalg.norm(action)

		energy_loss = action_norm * self.motor_energy_cost

		return action, energy_loss, motor_state, {"action_norm": action_norm}

	# ------------------------------------------------------------------

	def _decode_direct(self, neural_state: NeuralState, motor_state: MotorState) -> tuple[Action, Float, MotorState, Info]:

		action = neural_state.v[-2:]
		action = jnp.clip(action*self.wheel_speed_gain, -self.max_wheel_speed, self.max_wheel_speed).astype(jnp.float16)
		action_norm = jnp.linalg.norm(action)
		energy_loss = action_norm * self.motor_energy_cost
		return action, energy_loss, None, {"action_norm": action_norm}

	# ------------------------------------------------------------------

	def move(self, action: Action, body: Body) -> Body:
		# ---
		pos_dtype = body.pos.dtype
		if action.dtype != pos_dtype:
			print(f"action and position have different dtypes: {action.dtype} and {pos_dtype}")
			action = jnp.astype(action, pos_dtype)
		# ---
		
		radius = body.size/2

		def _if_not_equal(pos, heading, vr, vl):
			l = radius*2
			r = radius * (vl+vr) / (vr-vl)
			icc = pos + jnp.array([-r*jnp.sin(heading),r*jnp.cos(heading)], dtype=pos.dtype)
			omega = (vr-vl)/l
			omega = omega * self.dt
			rotation_matrix = jnp.array([[jnp.cos(omega), -jnp.sin(omega)],
										 [jnp.sin(omega),  jnp.cos(omega)]], dtype=pos.dtype)
			pos = rotation_matrix @ (pos-icc) + icc
			heading = heading + omega
			return Body(pos, heading, body.size)

		def _if_equal(pos, heading, vr, vl):
			pos = pos + vr*jnp.array([jnp.cos(heading), jnp.sin(heading)], dtype=pos.dtype) * self.dt
			return Body(pos, heading, body.size)

		pos, heading = body.pos, body.heading
		vl, vr = action
		is_equal = jnp.abs(vr-vl)<1e-5
		pos = jax.lax.cond(is_equal, _if_equal, _if_not_equal, pos, heading, vr, vl)
		return pos.replace(heading=jnp.mod(pos.heading, 2*jnp.pi))

	# ------------------------------------------------------------------

