from flax.struct import PyTreeNode
import jax
from jaxtyping import PyTree, Bool, Int16, UInt16, UInt32, Float16, Float

type NeuralState=PyTree
type NeuralParams=PyTree
type NeuralInput=PyTree
type Action=PyTree
type SensoryState=PyTree
type MotorState=PyTree
type Info=dict


class AgentConfig(PyTreeNode):
	# ------------------------------------------------------------------
	max_age: int
	init_energy: Float16
	max_energy: Float16
	basal_energy_loss: Float16
	size_energy_cost: Float16
	min_body_size: Float16=1.0
	max_body_size: Float16=10.0
	body_resolution: int|None=None
	time_below_threshold_to_die: int=30         # Time steps below energy threshold before death
	time_above_threshold_to_reproduce: int=100  # Time steps above energy threshold needed to reproduce
	reproduction_energy_cost: Float16=0.5
	# ------------------------------------------------------------------

class Genotype(PyTreeNode):
	# ------------------------------------------------------------------
	neural_params: NeuralParams
	body_size: Float
	chemical_emission_signature: jax.Array
	# ------------------------------------------------------------------

class Body(PyTreeNode):
	# ------------------------------------------------------------------
	pos: jax.Array
	heading: Float
	size: Float
	# ------------------------------------------------------------------

class AgentState(PyTreeNode):
	# ------------------------------------------------------------------
	genotype: Genotype
	# ---
	body: Body
	# ---
	motor_state: MotorState
	sensory_state: SensoryState
	neural_state: NeuralState
	# ---
	alive: Bool
	age: UInt16
	energy: Float16
	time_above_threshold: Int16
	time_below_threshold: Int16
	# --- infos
	n_offsprings: UInt16
	generation: UInt32
	id_: UInt32
	parent_id_: UInt32
	# ------------------------------------------------------------------

class Observation(PyTreeNode):
	# ------------------------------------------------------------------
	env: jax.Array
	internal: jax.Array
	# ------------------------------------------------------------------