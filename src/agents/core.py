from flax.struct import PyTreeNode
import jax

from typing import Callable, Tuple
import jax, jax.random as jr, jax.numpy as jnp
from flax.struct import PyTreeNode
from jaxtyping import PyTree, Bool, Int16, UInt16, UInt32, Float16, Float
import equinox as eqx
from functools import partial

type PolicyState=PyTree
type PolicyParams=PyTree
type PolicyInput=PyTree
type Action=PyTree
type SensoryState=PyTree
type MotorState=PyTree
type Observation=PyTree
type Info=dict

class Position(PyTreeNode):
	pos: jax.Array
	heading: jax.Array

class AgentState(PyTreeNode):
	# --- 
	policy_params: PolicyParams
	policy_state: PolicyState
	# ---
	position: Position
	# ---
	motor_state: MotorState
	sensory_state: SensoryState
	# ---
	alive: Bool
	age: Int16
	energy: Float16
	time_above_threshold: Int16
	time_below_threshold: Int16
	# ---
	reward: Float16
	reproduce: Bool
	# --- infos
	n_offsprings: UInt16
	generation: UInt32
	id_: UInt32
	parent_id_: UInt32