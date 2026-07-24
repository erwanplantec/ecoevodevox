import jax
from jax import numpy as jnp
from flax import struct
from jaxtyping import Float

from .core import MotorInterface, Action, Body, NeuralState, MotorState, Info
from ...settings import POSITION_DTYPE


def _scalar_motor(neural_state: NeuralState) -> jax.Array:
	"""Motor expression as one scalar per neuron.

	Grown networks may carry motor expression with a gene axis (RAND grows `m` as
	[N, motor_genes]); collapse it so the per-neuron masks below stay [N] rather than silently
	broadcasting to [N, N]. Networks whose `m` is already [N] (e.g. NeuronNCA) pass through.
	"""
	m = neural_state.m
	return m.mean(-1) if m.ndim == 2 else m


def outward_normals(xs: jax.Array, p: float, eps: float = 1e-6) -> jax.Array:
	"""Outward unit normals of the superellipse ``|x|^p + |y|^p = 1``, evaluated at `xs` [N, 2].

	The normal is the gradient of the shape function. For large `p` the superellipse tends to a
	square, so at a face centre this returns exactly the axis-aligned normal, while near a corner
	it blends smoothly between the two adjacent face normals.

	That smoothing is the point. Assigning each cilium to a face by dominant axis (as the original
	`ciliated` interface does) makes the normal jump across the diagonals: at (0.99, 0.98) it is
	(1,0) and at (0.98, 0.99) it is (0,1), which flips the sign of the torque for a hair's
	difference in neuron position. Development would then be unable to tune turning gradually.
	"""
	g = jnp.sign(xs) * jnp.abs(xs) ** (p - 1.0)
	return g / (jnp.linalg.norm(g, axis=-1, keepdims=True) + eps)


def make_ciliated_torque_morphogen_field(motor_interface, base_field=None):
	"""Build a RAND `fixed_morphogen_field` that tells cells the body's local surface geometry.

	Returns a callable ``field(x) -> [k + 2]`` suitable for `RAND_CTRNN(fixed_morphogen_field=...)`.
	It extends `base_field` (RAND's default positional morphogen `M` when None) with **two extra
	components: the outward unit normal of the ciliated_torque body at the cell's position** — but
	only for cells on the border (``max(|x|, |y|) > 1 - border_size``, the same test the interface
	uses to pick cilia). Off the border the two components are zero.

	The point is to give development the information that makes ciliated_torque locomotion work:
	a border cell can read which way it faces (hence whether beating there pushes forward, sideways,
	or turns the body) rather than having to infer it from raw position. `border_size` and `shape_p`
	are read from the interface so the morphogen's "on border" and normal direction match the
	locomotion exactly.

	Works on a single position ``x`` [2] (as RAND calls it, per cell) and on a batch [N, 2].
	"""
	if not hasattr(motor_interface, "shape_p"):
		raise TypeError("make_ciliated_torque_morphogen_field expects a CiliatedTorqueMotorInterface "
		                "(needs shape_p / border_size)")
	if base_field is None:
		from ..nn.rand import M            # local import avoids a motor<->nn import cycle
		base_field = M
	border_size = float(motor_interface.border_size)
	shape_p = float(motor_interface.shape_p)

	def field(x: jax.Array) -> jax.Array:
		on_border = jnp.maximum(jnp.abs(x[..., 0]), jnp.abs(x[..., 1])) > (1.0 - border_size)
		# `outward_normals` has a non-finite gradient at the origin (it divides by |x|), and
		# `jnp.where` leaks a NaN gradient from the *unselected* branch. RAND's migration
		# differentiates this field and interior cells sit at the origin, so evaluate the normal at
		# a safe off-origin point for non-border cells — their value is 0 either way, but this keeps
		# the gradient finite. Border cells (max(|x|) > 1-border_size) are never at the origin.
		safe = jnp.where(on_border[..., None], x, jnp.ones_like(x))
		normal = jnp.where(on_border[..., None], outward_normals(safe, shape_p), 0.0)
		return jnp.concatenate([base_field(x), normal], axis=-1)

	return field


class CiliatedTorqueMotorState(struct.PyTreeNode):
	"""Per-neuron cilia layout, precomputed once for a grown body.

	The body is rigid, so a cilium's geometric contribution per unit of beating never changes;
	only its beat rate does. Fields are [N] (or [N, 2]) over the network's neurons:
		cilium_size: motor expression read as cilium size (float, 0 for non-cilia), so thrust
		             scales continuously with how strongly the gene is expressed
		is_cilium:   motor neuron sitting on the body border
		thrust:      body-frame force per unit beat, ``-normal``
		lever:       torque per unit beat, ``cross(position, thrust)``
	"""
	cilium_size: jax.Array
	is_cilium: jax.Array
	thrust: jax.Array
	lever: jax.Array


class CiliatedTorqueMotorInterface(MotorInterface):
	"""Ciliated locomotion where every cilium both propels and steers.

	Unlike `ciliated`, which sorts border cilia into *either* velocity *or* turning by dominant
	axis, here each cilium exerts a force along the inward normal of the body surface and the
	resulting torque follows from rigid-body mechanics:

	    F_i   = -beat_i * n_i          (body-frame force, unchanged by where the cilium sits)
	    tau_i = x_i x F_i              (torque, proportional to offset from the face centre)

	So a cilium at the centre of a face contributes pure translation away from that face, and one
	toward the face edge contributes the *same* translation plus a turning moment that grows with
	its offset. Note the translation does not diminish toward the edge — the torque is added on
	top, not traded against thrust. Treating it as a trade-off would let a single corner cilium
	spin the body in place without translating it, which a single force cannot do.

	Motion is **holonomic**: side cilia push the body sideways, so the action is a body-frame
	velocity vector plus an angular speed, ``[vx, vy, omega]``, where ``+y`` is the heading
	direction and ``+x`` is starboard (matching how `AgentInterface.get_body_points` lays a body
	out). Locomotion is overdamped — force maps straight to velocity with no inertia — which is
	the correct regime for a ciliate rather than a simplification.

	Behaviour is independent of body size, as in `ciliated`: neuron positions are normalised to
	[-1, 1], so a bigger body has proportionally longer lever arms but they cancel. Physically,
	linear and angular drag scale differently with size (~size and ~size^3), so add explicit size
	terms here if that distinction matters for an experiment.

	Requires a spatially-embedded network state exposing `x` and `v` (and optionally `m`, `mask`).
	"""
	# ------------------------------------------------------------------
	dt: float = 1.0
	border_size: float = 0.2            # cilia sit where max(|x|,|y|) > 1 - border_size
	# Superellipse exponent, i.e. how square the body is. It trades steering authority at the
	# corners against continuity, and the trade is physical rather than numerical:
	#   p -> inf (square): torque grows linearly with offset all the way to the corner (maximum
	#     steering authority), but the corner is a true singularity — a cilium a hair onto the
	#     right face pushes left while one a hair onto the top face pushes down, so the torque
	#     flips sign across the diagonal and development cannot tune turning gradually there.
	#   small p (round): the normal at a corner points radially, straight through the body centre,
	#     so corner cilia produce *no* torque at all and are pure translators.
	# 16 keeps torque near-linear in offset over ~82% of each face while cutting the corner
	# discontinuity to ~1% of full scale.
	shape_p: float = 16.0
	thrust_gain: float = 1.0
	max_beat: float = 1.0
	max_velocity: float = 10.0
	max_angular_speed: float = jnp.pi / 4
	motor_energy_cost: float = 0.1
	# --- allometry (size/speed/energy trade-off). Neuron positions are normalised to [-1,1], so
	# the interface is size-independent by default (both exponents 0). Set them to couple body
	# size to locomotion:
	#   absolute velocity  *= body_size ** size_speed_exponent   (>0 -> bigger body absolutely
	#       faster in world units, but slower in body-lengths, as in real animals)
	#   motor energy cost  *= body_size ** size_cost_exponent    (>0 -> moving a bigger body costs
	#       more; the work-against-drag term)
	# A size-1 body is unaffected whatever the exponents; tune the base max_velocity / thrust_gain
	# / motor_energy_cost around the typical size when you turn these on.
	size_speed_exponent: float = 0.0
	size_cost_exponent: float = 0.0
	# ------------------------------------------------------------------

	def _cilia_layout(self, neural_state: NeuralState) -> CiliatedTorqueMotorState:
		assert hasattr(neural_state, "x"), \
			"ciliated_torque requires a spatially-embedded network (neural_state.x)"
		xs = neural_state.x

		# the body itself is a square (get_body_points samples a square grid), so "outer layer"
		# stays an L-infinity test; the superellipse is used only to smooth the thrust *direction*
		on_border = jnp.maximum(jnp.abs(xs[:, 0]), jnp.abs(xs[:, 1])) > (1.0 - self.border_size)

		# motor gene read as cilium size: a bigger cilium produces proportionally more thrust. No
		# expression threshold — size scales thrust continuously, so m -> 0 fades a cilium out
		# rather than switching it off at a cutoff. Clamped at 0: a negative size would invert
		# the thrust, giving a cilium that pulls the body toward the face it sits on.
		if hasattr(neural_state, "m"):
			cilium_size = jnp.maximum(_scalar_motor(neural_state), 0.0)
		else:
			# no motor gene means no size information: treat every cilium as unit-sized
			cilium_size = jnp.ones(xs.shape[0])

		is_cilium = on_border
		if getattr(neural_state, "mask", None) is not None:
			is_cilium = is_cilium & neural_state.mask.astype(bool)
		n = outward_normals(xs, self.shape_p)
		thrust = -n                                     # beating pushes the body away from the face
		# scalar z-torque of a unit beat: cross(x, thrust)
		lever = xs[:, 0] * thrust[:, 1] - xs[:, 1] * thrust[:, 0]

		return CiliatedTorqueMotorState(cilium_size=cilium_size * is_cilium, is_cilium=is_cilium,
		                                thrust=thrust, lever=lever)

	# ------------------------------------------------------------------

	def init(self, neural_state: NeuralState, *, key: jax.Array) -> CiliatedTorqueMotorState:
		return self._cilia_layout(neural_state)

	# ------------------------------------------------------------------

	def actuate(self, neural_state: NeuralState, motor_state: CiliatedTorqueMotorState,
	            body: Body, key: jax.Array) -> tuple[Body, Float, CiliatedTorqueMotorState, Info]:

		# Thrust per neuron: beat rate (non-negative — a cilium can stop, not suck) times cilium
		# size. `max_beat` caps the *rate*, so size scales the thrust that beating produces;
		# folding size in before the clip would let the cap swallow it (see `ciliated`).
		# `cilium_size` is already zero off the border, so non-cilia contribute nothing.
		beat = jnp.clip(neural_state.v * self.thrust_gain, 0.0, self.max_beat) * motor_state.cilium_size

		force = jnp.sum(motor_state.thrust * beat[:, None], axis=0)   # [2] body frame
		torque = jnp.sum(motor_state.lever * beat)

		# clip the speed's magnitude, not its components: clipping vx and vy separately would
		# rotate the velocity toward the diagonal instead of just shortening it
		speed = jnp.linalg.norm(force)
		velocity = force * jnp.where(speed > self.max_velocity, self.max_velocity / (speed + 1e-9), 1.0)
		omega = jnp.clip(torque, -self.max_angular_speed, self.max_angular_speed)

		# allometry: bigger body -> absolutely faster (velocity scales with size). Angular speed
		# is left size-independent. size**0 == 1, so the default (exponent 0) is unchanged.
		L = body.size.astype(POSITION_DTYPE)
		velocity = velocity * L ** self.size_speed_exponent

		# actions feed straight into positions, so build them at the position dtype
		action = jnp.array([velocity[0], velocity[1], omega], dtype=POSITION_DTYPE)
		new_body = self.move(action, body)

		# allometry: moving a bigger body costs more (work against drag)
		energy_loss = jnp.astype(jnp.sum(beat) * self.motor_energy_cost * L ** self.size_cost_exponent, jnp.float16)
		return new_body, energy_loss, motor_state, {"action_norm": jnp.abs(action).sum(), "action": action}

	# ------------------------------------------------------------------

	def move(self, action: Action, body: Body) -> Body:
		# ---
		pos_dtype = body.pos.dtype
		if action.dtype != pos_dtype:
			action = jnp.astype(action, pos_dtype)
		# ---
		vx, vy, omega = action
		# body -> world. The same rotation `get_body_points` uses (heading - pi/2), so the body
		# frame here is exactly the frame neuron positions live in: +y forward, +x starboard.
		a = body.heading - jnp.pi / 2
		ca, sa = jnp.cos(a), jnp.sin(a)
		world = jnp.array([ca * vx - sa * vy, sa * vx + ca * vy], dtype=pos_dtype)

		new_pos = body.pos + world * self.dt
		new_heading = jnp.mod(body.heading + omega * self.dt, 2 * jnp.pi)
		return body.replace(pos=new_pos, heading=new_heading)

	# ------------------------------------------------------------------
