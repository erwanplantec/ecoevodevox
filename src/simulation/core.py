from ..eco.gridworld import EnvState
from ..devo.core import AgentState

from flax.struct import PyTreeNode
import jax
from jaxtyping import Float, UInt32
from typing import Literal

class SimulationConfig(PyTreeNode):
    # ---
    max_agents: int=10_000  # Maximum number of agents that can exist simultaneously
    init_agents: int=1_024  # Initial number of agents at environment start
    birth_pool_size: int=256
    # ---
    wall_effect: Literal["kill","penalize","none"]="kill"  # What happens when agents hit walls
    wall_penalty: float=1.0                                # Penalty for hitting walls (if wall_effect==penalize)



class SimulationState(PyTreeNode):
    # ---
    env_state: EnvState
    agents_states: AgentState
    time: UInt32
    # ---
