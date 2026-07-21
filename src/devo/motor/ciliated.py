import jax
from jax import numpy as jnp
from flax import struct
from jaxtyping import Float

from .core import MotorInterface, Action, Body, NeuralState, MotorState, Info
from ...settings import POSITION_DTYPE


def _scalar_motor(neural_state: NeuralState) -> jax.Array:
	"""Motor expression as one scalar per neuron.

	Cilia use a scalar motor expression per neuron, but grown networks may carry it with a
	gene axis (RAND grows `m` as [N, motor_genes]); collapse that axis so the `& mask` below
	stays [N] rather than silently broadcasting to [N, N]. Networks whose `m` is already [N]
	(e.g. NeuronNCA) pass through.
	"""
	m = neural_state.m
	return m.mean(-1) if m.ndim == 2 else m


class CiliatedMotorState(struct.PyTreeNode):
	"""Precomputed, per-neuron cilia layout for a grown body.

	Fields are all shape ``[N]`` over the network's neurons:
		cilium_size:        motor expression read as cilium size (float, 0 for non-cilia), so
		                    thrust scales continuously with how strongly the gene is expressed
		is_velocity_cilium: motor neuron on the front/back border (drives velocity)
		is_angular_cilium:  motor neuron on the left/right border (drives turning)
		sign_x:             sign of the left/right coordinate  (right=+1, left=-1)
		sign_y:             sign of the front/back coordinate  (front=+1, back=-1)
	"""
	cilium_size: jax.Array
	is_velocity_cilium: jax.Array
	is_angular_cilium: jax.Array
	sign_x: jax.Array
	sign_y: jax.Array


class CiliatedMotorInterface(MotorInterface):
	"""Ciliated locomotion for a spatially-embedded (grown) network.

	Every neuron on the body **border** (``max(|x|) > 1 - border_size``) is a **cilium**, with
	its motor gene ``m`` read as the cilium's **size**. Thrust is the beating rate (activation
	``v`` times ``thrust_gain``, clipped to ``[0, max_beat]``) scaled by that size, so expression
	maps continuously to propulsion and ``m = 0`` simply means no cilium. Networks with no motor
	gene are treated as uniformly unit-sized.

	Cilia are split by which body edge they sit on (dominant coordinate axis):
		* front / back edges set the **linear velocity**
			- cilia on the **front** create **backward** thrust
			- cilia on the **back** create **forward** thrust
		* left / right edges set the **angular velocity** (omega, +ve = turn left / CCW)
			- cilia on the **right** turn the body **left**  (+omega)
			- cilia on the **left**  turn the body **right** (-omega)

	Action is ``[velocity, omega]``; ``move`` translates along the current heading and
	rotates in place. Requires a neural state exposing spatial fields ``x`` and ``v``
	(and optionally ``m`` / ``mask``), i.e. a developmental network such as ``NeuronNCA``.
	"""
	# ------------------------------------------------------------------
	dt: float = 1.0
	border_size: float = 0.2
	thrust_gain: float = 1.0
	max_beat: float = 1.0
	max_velocity: float = 10.0
	max_angular_speed: float = jnp.pi / 4
	motor_energy_cost: float = 0.1
	# ------------------------------------------------------------------

	def _cilia_layout(self, neural_state: NeuralState) -> CiliatedMotorState:
		"""Static (activation-independent) classification of neurons into cilia."""
		assert hasattr(neural_state, "x"), "ciliated interface requires a spatially-embedded network (neural_state.x)"
		xs = neural_state.x
		abs_x = jnp.abs(xs[:, 0])
		abs_y = jnp.abs(xs[:, 1])

		threshold = 1.0 - self.border_size
		on_border = jnp.maximum(abs_x, abs_y) > threshold

		# motor gene read as cilium size. No expression threshold: size scales thrust continuously,
		# so m -> 0 fades a cilium out smoothly instead of switching it off at a cutoff (a step
		# there would sit exactly where development is tuning). Clamped at 0 because the threshold
		# used to guarantee positivity, and a negative size would mean a cilium that sucks.
		if hasattr(neural_state, "m"):
			cilium_size = jnp.maximum(_scalar_motor(neural_state), 0.0)
		else:
			# a network without a motor gene has no size information: treat every cilium as
			# unit-sized. Leaving cilium_size unset here raised UnboundLocalError below.
			cilium_size = jnp.ones(xs.shape[0])

		is_cilium = on_border
		# respect the developmental mask (dead / absent neurons) when present
		if hasattr(neural_state, "mask") and neural_state.mask is not None:
			is_cilium = is_cilium & neural_state.mask.astype(bool)
		cilium_size = cilium_size * is_cilium
		# split border cilia by dominant axis -> side (left/right) vs front/back
		on_side = abs_x >= abs_y
		is_angular_cilium = is_cilium & on_side       # left / right edges
		is_velocity_cilium = is_cilium & (~on_side)   # front / back edges

		return CiliatedMotorState(
			cilium_size=cilium_size,
			is_velocity_cilium=is_velocity_cilium,
			is_angular_cilium=is_angular_cilium,
			sign_x=jnp.sign(xs[:, 0]),
			sign_y=jnp.sign(xs[:, 1]),
		)

	# ------------------------------------------------------------------

	def init(self, neural_state: NeuralState, *, key: jax.Array) -> CiliatedMotorState:
		return self._cilia_layout(neural_state)

	# ------------------------------------------------------------------

	def decode(self, neural_state: NeuralState, motor_state: CiliatedMotorState) -> tuple[Action, Float, CiliatedMotorState, Info]:

		# Thrust per neuron: beat rate (non-negative, gated by activation) times cilium size.
		# `max_beat` caps how fast a cilium can beat, so it applies to the rate; size then scales
		# the thrust that beating produces. Folding size in *before* the clip instead lets the cap
		# swallow it — with the tuned config ~79% of cilia sit at max_beat, and thrust comes out
		# uncorrelated with size (measured r = -0.01 over 965 samples, vs +0.80 this way), i.e.
		# "bigger cilium = more thrust" silently stops holding exactly when pushing hardest.
		beat = jnp.clip(neural_state.v * self.thrust_gain, 0.0, self.max_beat) * motor_state.cilium_size

		# front/back cilia -> velocity. front (sign_y>0) pushes backward, back pushes forward.
		velocity_contrib = jnp.where(motor_state.is_velocity_cilium, -motor_state.sign_y * beat, 0.0)
		velocity = jnp.clip(jnp.sum(velocity_contrib), -self.max_velocity, self.max_velocity)

		# left/right cilia -> omega. right (sign_x>0) turns left (+omega), left turns right.
		omega_contrib = jnp.where(motor_state.is_angular_cilium, motor_state.sign_x * beat, 0.0)
		omega = jnp.clip(jnp.sum(omega_contrib), -self.max_angular_speed, self.max_angular_speed)

		# actions feed straight into positions, so keep them at the position dtype: building
		# them in float16 and upcasting in move() would quantise the step for no benefit
		action = jnp.array([velocity, omega], dtype=POSITION_DTYPE)

		active_beat = jnp.where(motor_state.is_velocity_cilium | motor_state.is_angular_cilium, beat, 0.0)
		energy_loss = jnp.astype(jnp.sum(active_beat) * self.motor_energy_cost, jnp.float16)
		action_norm = jnp.abs(action).sum()

		return action, energy_loss, motor_state, {"action_norm": action_norm}

	# ------------------------------------------------------------------

	def move(self, action: Action, body: Body) -> Body:
		# ---
		pos_dtype = body.pos.dtype
		if action.dtype != pos_dtype:
			action = jnp.astype(action, pos_dtype)
		# ---
		velocity, omega = action
		direction = jnp.array([jnp.cos(body.heading), jnp.sin(body.heading)], dtype=pos_dtype)
		new_pos = body.pos + velocity * self.dt * direction
		new_heading = jnp.mod(body.heading + omega * self.dt, 2 * jnp.pi)
		return body.replace(pos=new_pos, heading=new_heading)

	# ------------------------------------------------------------------
