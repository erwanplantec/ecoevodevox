# from flax.struct import PyTreeNode
# import jax
# from jax import numpy as jnp, random as jr, nn as jnn
# import equinox as eqx
# import equinox.nn as nn
# from flax import struct
# from jaxtyping import Float
# import numpy as np

# from .core import MotorInterface, Action, Body, NNState, MotorState, Info


# class CiliatedMotorState(struct.PyTreeNode):
# 	is_motor: Float
# 	on_side: Float

# class CiliatedMotorInterface(MotorInterface):
# 	#-------------------------------------------------------------------
# 	dt: float=1.0
# 	border_size: float=0.2
# 	max_neuron_force: float=0.2
# 	neuron_force_gain: float=0.2
# 	max_angular_speed: float=jnp.pi/4
# 	max_velocity: float=10.0
# 	motor_energy_cost: float=0.1
# 	motor_expression_threshold: float=0.9
# 	#-------------------------------------------------------------------

# 	def init(self, policy_state: NNState, key: jax.Array) -> CiliatedMotorState:
# 		# ---
# 		assert hasattr(policy_state, "x") #make sure network is spatially embedded
# 		assert hasattr(policy_state, "m") 
# 		# ---
# 		xs = policy_state.x
# 		threshold = 1-self.border_size
# 		is_motor = (policy_state.m[:,0] > self.motor_expression_threshold) & (jnp.abs(xs).max(-1) > threshold)
# 		side_motors = is_motor & (jnp.abs(xs[:,0]) > threshold)

# 		return CiliatedMotorState(is_motor, side_motors)

# 	#-------------------------------------------------------------------

# 	def decode(self, policy_state: NNState, motor_state: CiliatedMotorState) -> tuple[Action, Float, CiliatedMotorState, Info]:
		
# 		xs = policy_state.x
# 		ms = jnp.clip(policy_state.m[:,0], min=0.0)
# 		vs = policy_state.v

# 		forces = jnp.where(motor_state.is_motor, vs * ms * self.neuron_force_gain, 0.0); assert isinstance(forces, jax.Array)
# 		forces = jnp.clip(forces, 0, self.max_neuron_force)
# 		omegas = jnp.where(
# 			motor_state.on_side, 
# 			forces * xs[:,1] * jnp.sign(xs[:,0]),
# 			xs[:,0] * forces * (-jnp.sign(xs[:,1]))
# 		)
# 		omega = jnp.clip(jnp.sum(omegas), -self.max_angular_speed, self.max_angular_speed)
# 		velocity = jnp.clip(
# 			jnp.sum(jnp.clip(jnp.where(motor_state.on_side, 0.0, -forces*jnp.sign(xs[:,0])))),
# 			-self.max_velocity, self.max_velocity
# 		)
# 		action = jnp.array([omega, velocity])

# 		energy_loss = jnp.sum(forces * self.motor_energy_cost)

# 		return action, energy_loss, motor_state, {"action_norm": energy_loss}

# 	#-------------------------------------------------------------------
	
# 	def move(self, action: Action, body: Body) -> Body:
		
# 		omega, vel = action

# 		new_heading = jnp.mod(body.heading + omega, 2*jnp.pi)
# 		new_pos = body.pos + vel*jnp.array([jnp.cos(body.heading), jnp.sin(body.heading)])

# 		return body.replace(pos=new_pos, heading=new_heading)
# 	#-------------------------------------------------------------------

