import jax
from jax import numpy as jnp, random as jr, nn as jnn
from jax.experimental import io_callback
from jax.sharding import PartitionSpec as P, NamedSharding
import equinox as eqx
import equinox.nn as nn
import wandb
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import yaml
from functools import partial
from typing import Callable
import random
import string
import datetime

from .agents.motor.base import MotorInterface
from .agents.nn.base import Policy
from .agents.sensory.base import SensoryInterface
from .evo.core import MutationModel
from .agents.core import AgentState, Genotype
from .eco.gridworld import EnvState, FoodType, ChemicalType, GridWorld, GridworldConfig
from .agents import AgentInterface
from .agents.motor import motor_interfaces
from .agents.sensory import sensory_interfaces
from .agents.nn import make_apply_init, nn_models
from .devo import encoding_models
from .evo import mutation_models

def generate_random_name(length=8):
	return "".join([random.choice(string.ascii_lowercase+string.digits) for _ in range(length)])

def load_config(filename):
	with open(filename, "r") as file:
		cfg = yaml.safe_load(file)
	return cfg

#=======================================================================

def metrics_fn(state: EnvState, step_data: dict)->dict:
	"""computes metrics from world state and step_data to log"""
	agents = state.agents_states
	food_levels = {
		f"food_{i}": state.food[i].sum() for i in range(state.food.shape[0])
	}
	data = {
		"alive": agents.alive,
		"population": agents.alive.sum(),
		"energy_levels": agents.energy,
		"offsprings": agents.n_offsprings,
		**step_data,
		"ages": agents.age,
		**food_levels,
		"total_food": state.food.sum(),
		"network_sizes": agents.policy_state.mask.sum(-1)
	}
	return data

def host_log_transform(data: dict)->dict:
	data = jax.tree.map(np.asarray, data)
	mask = data["alive"]
	avg_data = {}
	for k, v in data.items():
		if k=="alive" or not v.shape:
			continue
		if v.shape[0]==mask.shape[0]:
			data[k] = v[mask]
			avg_data[f"{k} (avg)"] = np.mean(data[k])
	data["reproduction_rates"] = data["offsprings"] / data["ages"]
	del data["alive"] 
	data = {**data, **avg_data}
	return data

#=======================================================================

def init_agents_interface(cfg: dict, key: jax.Array)->tuple[AgentInterface, MutationModel]:
	"""initializes the agents interface and the mutation function"""
	#---
	motor_cfg = cfg["agents"]["motor"]
	motor_cls = motor_interfaces.get(motor_cfg["which"],None); assert motor_cls is not None, f"motor interface {motor_cfg['which']} is not valid"
	motor_kwargs = {k:v for k,v in motor_cfg.items() if k !="which"}
	motor_interface: MotorInterface = motor_cls(**motor_kwargs)
	#---
	sensory_cfg = cfg["agents"]["sensory"]
	sensory_cls = sensory_interfaces.get(sensory_cfg["which"], None); assert sensory_cls is not None, f"sensory interface {sensory_cfg['which']} is not valid"
	sensory_kwargs = {k:v for k,v in sensory_cfg.items() if k !="which"}
	sensory_interface: SensoryInterface = sensory_cls(**sensory_kwargs)
	#---
	enc_cfg = cfg["agents"]["encoding"]
	enc_cls = encoding_models.get(enc_cfg['which'], None); assert enc_cls is not None, f"encoding model {enc_cfg['which']} is not valid"
	enc_kwargs = {k:v for k,v in enc_cfg.items() if k !="which"}
	encoding_fctry = lambda key: enc_cls(**enc_kwargs, key=key)
	#---
	nn_cfg = cfg["agents"]["nn"]
	nn_cls = nn_models.get(nn_cfg["which"], None); assert nn_cls is not None, f"nn model {nn_cfg['which']} is not valid"
	nn_kwargs = {k:v for k,v in nn_cfg.items() if k !="which"}
	policy_fctry = lambda key: nn_cls(encoding_model=encoding_fctry(key), **nn_kwargs)
	prms_fctry = lambda key: eqx.filter(policy_fctry(key), eqx.is_array)
	policy: Policy = policy_fctry(key)
	policy_prms, _ = eqx.partition(policy, eqx.is_array)
	policy_apply, policy_init = make_apply_init(policy, reshape_prms=False)
	# ---
	mut_cfg = cfg["agents"]["mutation"]
	cls = mutation_models.get(mut_cfg["which"], None); assert cls is not None, f"mutation mdl {mut_cfg['which']} is not valid"
	kwargs = {k:v for k,v in mut_cfg.items() if k !="which"}
	genotype_like = Genotype(policy_prms, jnp.asarray(0.0))
	mutation_fn: MutationModel = cls(genotype_like=genotype_like, **kwargs)
	# ---

	interface = AgentInterface(policy_apply=policy_apply,
							   policy_init=policy_init,
							   policy_fctry=prms_fctry,
							   sensory_interface=sensory_interface,
							   motor_interface=motor_interface,
							   body_resolution=cfg["agents"]["body_resolution"],
							   basal_energy_loss=cfg["agents"]["basal_energy_loss"])

	return interface, mutation_fn

#=======================================================================

def make_world(cfg: dict, key: jax.Array)->GridWorld:
	"""initializes the world"""

	env_cfg = cfg["env"]

	agents_interface, mutation_fn = init_agents_interface(cfg, key)

	cfg_ct = {k:v for k,v in cfg.items() if k.startswith("ct")}
	cfg_ft = {k:v for k,v in cfg.items() if k.startswith("ft")}

	chemical_types = jax.tree.map(
		lambda *ct: jnp.stack(ct), *[ct for ct in cfg_ct.values()]
	)
	chemical_types = ChemicalType(**chemical_types)

	for typ in cfg_ft.keys():
		if isinstance(cfg_ft[typ]["chemical_signature"], int):
			cfg_ft[typ]["chemical_signature"] = jnn.one_hot(cfg_ft[typ]["chemical_signature"], len(cfg_ct))
		elif isinstance(cfg_ft[typ]["chemical_signature"], int|tuple): 
			assert len(cfg_ft[typ]["chemical_signature"])==len(cfg_ct)
			cfg_ft[typ]["chemical_signature"] = jnp.asarray(cfg_ft[typ]["chemical_signature"])

	food_types = jax.tree.map(
		lambda *fts: jnp.stack([jnp.asarray(ft, dtype=jnp.float32) for ft in fts]), 
		*list(cfg_ft.values()),
		is_leaf=lambda x: isinstance(x, list)
	)
	food_types = FoodType(**food_types)

	world_cfg = GridworldConfig(**env_cfg)
	world = GridWorld(world_cfg, 
					  agent_interface=agents_interface, 
					  mutation_fn=mutation_fn, 
					  chemical_types=chemical_types, 
					  food_types=food_types, 
					  key=jr.key(cfg["seed"]))

	return world


#=======================================================================

class Simulator:
	#-------------------------------------------------------------------
	def __init__(self, 
				 world: GridWorld,  # The gridworld environment to simulate
				 key: jax.Array,  # Random key for simulation
				 log: bool=False,  # Whether to log metrics during simulation
				 name: str|None=None,
				 ckpt_freq: int|None=None,  # Frequency of checkpoint saves (in steps)
				 sampling_freq: int|None=None, # Frequency of sampling
				 sampling_size: int=16, # Size of samples
				 n_devices: int|None=None, # Number of devices to use (None for all available)
				 metrics_fn: Callable=metrics_fn, # Function to compute metrics
				 host_log_transform: Callable=host_log_transform # Function to transform metrics for logging
				):  
		# ---
		if name:
			self.name = name
		else:
			date = datetime.datetime.now()
			self.name = date.strftime("%d_%m_%Y_%H:%M:%S")
		# ---
		self.world = world
		key_sim, key_aux = jr.split(key)
		self.key_sim = key_sim
		self.key_aux = key_aux
		self.n_devices = jax.device_count() if n_devices is None else n_devices
		assert world.cfg.max_agents % self.n_devices == 0
		# ---
		self.log = log
		# ---
		self.ckpt_freq = ckpt_freq
		if ckpt_freq is not None and ckpt_freq>0:
			ckpt_dir = f"data/{self.name}/ckpts"
			os.makedirs(ckpt_dir, exist_ok=True)
		else:
			ckpt_dir = None
		# ---
		self.sampling_freq = sampling_freq
		self.sampling_size = sampling_size
		if sampling_freq is not None and sampling_freq>0:
			sampling_dir = f"data/{self.name}/samples"
			os.makedirs(sampling_dir, exist_ok=True)
		else:
			sampling_dir = None
		# ---

		def _log_clbk(data: dict):
			data = host_log_transform(data)
			wandb.log(data)
			return jnp.zeros((), dtype=bool)

		def _ckpt_clbk(state_dict):
			time = state_dict["env_state"].time
			filename = f"{ckpt_dir}/{int(time)}.pickle"
			with open(filename, "wb") as file:
				pickle.dump(state_dict, file)
			return jnp.zeros((),dtype=bool)

		def _sample_clbk(sample, time):
			filename = f"{sampling_dir}/{int(time)}.pickle"
			with open(filename, "wb") as file:
				pickle.dump(sample, file)
			return jnp.zeros((),dtype=bool)

			

		device_mesh = jax.make_mesh((self.n_devices,), ('N',))
		state_shardings = EnvState(
			agents_states=NamedSharding(device_mesh, P("N")), #type:ignore
			food=None,		   #type:ignore
			time=None,
			last_agent_id=None
		)

		@partial(jax.jit, out_shardings=state_shardings) #type:ignore
		def _initialize(key:jax.Array)->EnvState:
			return self.world.init(key)


		@partial(jax.jit, in_shardings=(state_shardings,None), out_shardings=state_shardings) #type:ignore
		def _simulation_step(state: EnvState, key: jax.Array)->EnvState:
			# ---
			key, key_step = jr.split(key)
			state, step_data = world.step(state, key_step)
			# ---
			if log:
				data = metrics_fn(state, step_data)
				io_callback(_log_clbk, jax.ShapeDtypeStruct((),bool), data)
			if ckpt_dir:
				assert isinstance(ckpt_freq, int)
				_ = jax.lax.cond(
					jnp.mod(state.time, ckpt_freq)==0,
					lambda s: io_callback(_ckpt_clbk, jax.ShapeDtypeStruct((),bool), s),
					lambda *a, **k: jnp.zeros((), dtype=bool),
					{"env_state": state, "key": key}
				)
			if sampling_dir:
				assert isinstance(sampling_freq, int)
				def _sample_and_clbk(agents, time, key):
					p = agents.alive / agents.alive.sum()
					sample_ids = jr.choice(key, agents.alive.shape[0], shape=(sampling_size,), p=p)
					sample = jax.tree.map(lambda x: x[sample_ids], agents)
					return io_callback(_sample_clbk, jax.ShapeDtypeStruct((),bool), sample, time)
				_ = jax.lax.cond(
					jnp.mod(state.time, sampling_freq)==0,
					_sample_and_clbk,
					lambda *a, **k: jnp.zeros((), dtype=bool),
					state.agents_states, state.time, key
				)
			# ---
			return state

		self.simulation_step = _simulation_step

		@partial(jax.jit, static_argnames=("steps"))
		def _simulate(state: EnvState, key: jax.Array, steps: int)->EnvState:

			def _step(_, c):
				state, key = c
				key, _key = jr.split(key)
				state = _simulation_step(state, _key)
				return state, key

			state, _ = jax.lax.fori_loop(
				0, steps, _step, (state, key)
			)

			return state

		# ---

		self.initialize = _initialize
		self.simulate = _simulate

		# ---

		self.world_state = None

	#-------------------------------------------------------------------

	def run_interactive(self):
		
		key_sim, key_aux = self.key_sim, self.key_aux
		world_state = self.world_state

		if world_state is None:
			print("initializing world ...")
			key_sim, key_init = jr.split(key_sim)
			world_state = self.initialize(key_init)
			print("initialization completed")

		print(f"""
		Starting interactive simulation !

		Run name: {self.name}
		
		Commands:
		s, sim, simulate [steps]: simulate the world for a given number of steps (default: 1)
		r, render: render the world
		q: quit the simulation
		h, help: show this help message
		""")

		while True:
			user_input = input("cmd: ")
			cmd, *args = user_input.strip().split(" ")
			
			# ---

			if cmd in ["s", "sim", "simulate"]:
				if not args:
					steps = 1
				else:
					steps = int(args[0])
				key_sim, _key = jr.split(key_sim)
				world_state = self.simulate(world_state, _key, steps)

			# ---

			elif cmd in ["r", "render"]:
				ax = plt.figure().add_subplot()
				self.world.render(world_state, ax)
				plt.show()

			# ---

			elif cmd == "q":
				break

			# ---

			elif cmd in ["h", "help"]:
				print("""
					Commands:
					s, sim, simulate [steps]: simulate the world for a given number of steps (default: 1)
					r, render: render the world
					q: quit the simulation
					h, help: show this help message
				""")

			# ---

			else:
				print("invalid cmd, use q to exit")

			# ---

		self.world_state = world_state
		self.key_sim = key_sim
		self.key_aux = key_aux

	#-------------------------------------------------------------------
	
	def load_ckpt(self, filename: str):
		with open(filename, "rb") as file:
			state_dict = pickle.load(file)
		self.state = state_dict["env_state"]
		self.key_sim = state_dict["key"]

	#-------------------------------------------------------------------
	
	@classmethod
	def from_config_file(cls, filename: str):
		cfg = load_config(filename)
		key = jr.key(cfg["seed"])
		key, key_make = jr.split(key)
		world = make_world(cfg, key_make)
		return (
			cls(world, key, log=cfg["log"], ckpt_freq=cfg.get("ckpt_freq", None), sampling_freq=cfg.get("sampling_freq", None), sampling_size=cfg.get("sampling_size", 16)), 
			cfg
		)

	#-------------------------------------------------------------------

class ScenarioSimulator():
	#-------------------------------------------------------------------
	def __init__(
		self, 
		meta_cfg_file: str, 
		log: bool=False,
	 	ckpt_dir: str|None=None,
	 	ckpt_freq: int=10_000, 
	 	n_devices: int|None=None):


		cfg = load_config(meta_cfg_file)
		simulators = []
		key = jr.key(cfg["seed"])

		
		for k, value in cfg.items():

			if k.startswith("env"):
				key, k1, k2 = jr.split(key, 3)
				cfg_file = value["file"]
				cfg = load_config(cfg_file)
				wrld = make_world(cfg, k1)
				sim = Simulator(wrld, k2, log, ckpt_dir, ckpt_freq, n_devices)

				_init_food = jax.jit(wrld._init_food, out_shardings=None) #type:ignore

				def _step(state: EnvState, key: jax.Array)->EnvState:
					if value["on_start"]=="init":
						key, _key = jr.split(key)
						state = jax.lax.cond(
							state.time==value["from"],
							lambda s, k: s.replace(food=s.food|_init_food(k)),
							lambda s, k: s,
							state, _key
						)
					state = sim.simulation_step(state, key)
					return state

				simulators.append((_step, value))

		
		key, key_init = jr.split(key)
		for x in simulators:
			print(x[1])
		simulators = sorted(simulators, key=lambda x: x[1]["from"])
		state = wrld.init(key_init) #type:ignore
		state = state.replace(food=jnp.zeros_like(state.food))
		self.state = state

		def _simulation_step(state: EnvState, key: jax.Array)->EnvState:
			
			done = jnp.asarray(False)
			for sim_fn, cfg in simulators:


				state, done = jax.lax.cond(
					(cfg["from"]<=state.time) & (state.time<cfg["to"]) & (~done),
					lambda s, k: (sim_fn(s,k), jnp.asarray(True)),
					lambda s, k: (s, done),
					state, key
				)

			return state

		self.simulation_step = _simulation_step


		@partial(jax.jit, static_argnames=("steps"))
		def _simulate(state: EnvState, key: jax.Array, steps: int)->EnvState:

			def _step(_, c):
				state, key = c
				key, _key = jr.split(key)
				state = _simulation_step(state, _key)
				return state, key

			state, _ = jax.lax.fori_loop(
				0, steps, _step, (state, key)
			)

			return state

		self.simulate = _simulate



if __name__ == '__main__':
	scenario_file = "configs/meta_config.yml"

	simulator = ScenarioSimulator(scenario_file)
	simulator.simulate(simulator.state, jr.key(1), 100)