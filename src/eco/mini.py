import jax
from jax import numpy as jnp, random as jr, nn as jnn
from flax.struct import PyTreeNode
from jaxtyping import Float

from ..agents.core import AgentState, PolicyParams, Genotype, Body
from ..agents.interface import AgentInterface
from ..agents.nn.base import Policy
from .gridworld import EnvState, get_cell_index, Observation

class MiniAgentState(AgentState):
	def __init__(self, genotype, body, motor_state, sensory_state, policy_state, **kwargs):
		super().__init__(genotype, body, motor_state, sensory_state, policy_state, 
			True, 1, 0, 0, 0, 0, False, 0, 0, 0, 0)


class MiniEnvState(PyTreeNode):
	state_grid: jax.Array
	agent_state: AgentState


class MiniEnv:
	#-------------------------------------------------------------------
	def __init__(self, grid_size: tuple[int,int], agent_interface: AgentInterface):
		self.grid_size = grid_size
		self.agent_interface = agent_interface
	#-------------------------------------------------------------------
	def reset(self, params: PolicyParams, key: jax.Array)->MiniEnvState:
		raise NotImplementedError
	#-------------------------------------------------------------------
	def init_agent_state(self, genotype: Genotype, key: jax.Array)->MiniAgentState:
		k1, k2 = jr.split(key)

		start_pos = jnp.asarray(self.grid_size)/2
		start_heading = jr.uniform(k1, (), minval=0, maxval=jnp.pi*2)
	
		policy_state, sensory_state, motor_state, body_size = self.agent_interface.init(genotype, k2)
		body = Body(start_pos, start_heading, body_size)
		return MiniAgentState(genotype, body, motor_state, sensory_state, policy_state)
	#-------------------------------------------------------------------
	def step(self, state: MiniEnvState, key: jax.Array)->MiniEnvState :
		obs = self.get_observation(state)
		action, agent_state, _ = self.agent_interface.step(obs, state.agent_state, key)
		new_body = self.agent_interface.move(action, agent_state.body)
		new_body = new_body.replace(
			pos = jnp.clip(new_body.pos, 0, jnp.asarray(self.grid_size)-0.0001)
		)
		agent_state = agent_state.replace(body=new_body)
		return state.replace(agent_state=agent_state)
	#-------------------------------------------------------------------
	def get_observation(self, state: MiniEnvState):
		body = state.agent_state.body
		indices = get_cell_index(self.agent_interface.full_body_pos(body))
		obs = state.state_grid[:,*indices]
		return Observation(obs, jnp.asarray(0.0), jnp.zeros((1, *obs.shape[1:])))
	#-------------------------------------------------------------------
	def rollout(self, params: PolicyParams, steps: int,  key: jax.Array)->MiniEnvState:
		def _step(state, key):
			new_state = self.step(state, key)
			return new_state, state
		key_init, key_roll = jr.split(key)
		state = self.reset(params, key_init)
		_, states = jax.lax.scan(_step, state, jr.split(key_roll, steps))
		return states
	#-------------------------------------------------------------------
	def evaluate(self, params: PolicyParams, key: jax.Array)->tuple[Float,dict]:
		raise NotImplementedError


class MoveTrivialEnv(MiniEnv):
	#-------------------------------------------------------------------
	def reset(self, params: PolicyParams, key: jax.Array) -> MiniEnvState:
		k1, k2 = jr.split(key)

		genotype = Genotype(params, 2.0)
		agent_state = self.init_agent_state(genotype, k1)

		grid = jnp.ones((1,*self.grid_size))

		return MiniEnvState(grid, agent_state)
	#-------------------------------------------------------------------
	def evaluate(self, params: PolicyParams, key: jax.Array) -> tuple[Float, dict]:
		state = self.rollout(params, 32, key)
		return jnp.linalg.norm(state.agent_state.body.pos), {}


class GatherState(MiniEnvState):
	food: jax.Array

diff_kernel = jnp.array([[0.1, 0.1, 0.1], [0.1, 1.0, 0.1], [0.1, 0.1, 0.1]])

class Gather(MiniEnv):
	#-------------------------------------------------------------------
	def __init__(self, grid_size: tuple[int, int], agent_interface: AgentInterface,
		density: float=1.0):
		super().__init__(grid_size, agent_interface)
		self.density=density
	#-------------------------------------------------------------------
	def reset(self, params: PolicyParams, key: jax.Array) -> GatherState:
		k1, k2 = jr.split(key)

		genotype = Genotype(params, 2.0)
		agent_state = self.init_agent_state(genotype, k1)

		food = jr.bernoulli(k2, self.density, self.grid_size)
		chems = jax.scipy.signal.convolve2d(food, diff_kernel)[None]

		return GatherState(chems, agent_state,food)
	#-------------------------------------------------------------------
	def step(self, state: GatherState, key: jax.Array) -> GatherState:
		state = super().step(state, key)
		bps = self.agent_interface.full_body_pos(state.agent_state.body)
		food = state.food.at[*get_cell_index(bps)].set(False)
		chems = jax.scipy.signal.convolve2d(food, diff_kernel)[None]
		return GatherState(chems, state.agent_state, food)
	#-------------------------------------------------------------------
	def evaluate(self, params: PolicyParams, key: jax.Array) -> tuple[Float, dict]:
		states: GatherState = self.rollout(params, 32, key)
		return -jnp.sum(states.food[-1]), {}


class ChemotaxisEnv(MiniEnv):
	#-------------------------------------------------------------------
	def __init__(self, interface: AgentInterface, sigma: float=5.0, grid_size: tuple[int,int]=(32,32),
		move_bonus: float=0.01):
		super().__init__(grid_size, interface)
		self.sigma = sigma
		self.move_bonus = move_bonus
	#-------------------------------------------------------------------
	def reset(self, params: PolicyParams, key: jax.Array)->MiniEnvState:

		k1, k2 = jr.split(key)

		genotype = Genotype(params, 2.0)
		agent_state = self.init_agent_state(genotype, k1)

		optima = jr.choice(
			k2, 
			jnp.array([
				[0,0],
				[0,self.grid_size[1]],
				[self.grid_size[0], 0],
				list(self.grid_size)
			])
		)
		grid = jnp.mgrid[:self.grid_size[0], :self.grid_size[1]]
		dists = jnp.linalg.norm(grid-optima[:,None,None], axis=0)
		chem_grid = jnp.exp(-dists**2/self.sigma)

		return MiniEnvState(chem_grid[None], agent_state)
	#-------------------------------------------------------------------
	def evaluate(self, params: PolicyParams, key: jax.Array) -> tuple[Float, dict]:
		states = self.rollout(params, 32, key)
		grid = states.state_grid[0,0]
		i, j = get_cell_index(states.agent_state.body.pos).T #T,2
		vals = grid[i,j]
		fitness = jnp.sum(vals)
		return fitness, dict()


def make(cfg, agent_interface):
	env = cfg["which"]
	if env=="chemotaxis":
		return ChemotaxisEnv(agent_interface, **{k:v for k,v in cfg.items() if k!="which"})
	elif env=="trivial":
		return MoveTrivialEnv((32,32), agent_interface)
	elif env=="gather":
		return Gather((32,32), agent_interface, density=cfg.get("density",1.0))
	else:
		raise KeyError(f"No env named {env}")





