from functools import partial
import yaml
import argparse
import jax
from jax import numpy as jnp, random as jr, nn as jnn
import equinox as eqx
import equinox.nn as nn
import numpy as np
from jax.experimental import io_callback
import pickle
import wandb
import os
import matplotlib.pyplot as plt

from src.eco.gridworld import EnvState, FoodType, ChemicalType, GridWorld, GridworldConfig
from src.agents import AgentInterface
from src.agents.motor import motor_interfaces
from src.agents.sensory import sensory_interfaces
from src.devo import encoding_models
from src.nn import make_apply_init, nn_models
from src.evo import mutation_models

#=======================================================================

def load_config(filename):
	with open(filename, "r") as file:
		cfg = yaml.safe_load(file)
	return cfg

#=======================================================================

def init_agents_interface(cfg: dict, key: jax.Array):
	""""""
	#---
	
	motor_cfg = cfg["agents"]["motor"]
	cls = motor_interfaces.get(motor_cfg["which"],None); assert cls is not None, f"motor interface {motor_cfg['which']} is not valid"
	kwargs = {k:v for k,v in motor_cfg.items() if k !="which"}
	motor_interface = cls(**kwargs)

	#---
	
	sensory_cfg = cfg["agents"]["sensory"]
	cls = sensory_interfaces.get(sensory_cfg["which"], None); assert cls is not None, f"sensory interface {sensory_cfg['which']} is not valid"
	kwargs = {k:v for k,v in sensory_cfg.items() if k !="which"}
	sensory_interface = cls(**kwargs)

	# ---

	enc_cfg = cfg["agents"]["encoding"]
	cls = encoding_models.get(enc_cfg['which'], None); assert cls is not None, f"encoding model {enc_cfg['which']} is not valid"
	kwargs = {k:v for k,v in enc_cfg.items() if k !="which"}
	encoding_model = cls(**kwargs, key=key)

	# ---

	nn_cfg = cfg["agents"]["nn"]
	cls = nn_models.get(nn_cfg["which"], None); assert cls is not None, f"nn model {nn_cfg['which']} is not valid"
	kwargs = {k:v for k,v in nn_cfg.items() if k !="which"}
	policy = cls(encoding_model=encoding_model, **kwargs)
	policy_prms, _ = eqx.partition(policy, eqx.is_array)
	policy_apply, policy_init = make_apply_init(policy, reshape_prms=False)

	# ---

	mut_cfg = cfg["agents"]["mutation"]
	cls = mutation_models.get(mut_cfg["which"], None); assert cls is not None, f"mutation mdl {mut_cfg['which']} is not valid"
	kwargs = {k:v for k,v in mut_cfg.items() if k !="which"}
	mutation_fn = cls(prms_like=policy_prms, **kwargs)

	# ---

	def policy_fctry(key: jax.Array):
		return mutation_fn(policy_prms, key)

	# ---

	interface = AgentInterface(policy_apply=policy_apply,
							   policy_init=policy_init,
							   policy_fctry=policy_fctry,
							   sensory_interface=sensory_interface,
							   motor_interface=motor_interface,
							   size=cfg["agents"]["size"],
							   body_resolution=cfg["agents"]["body_resolution"],
							   basal_energy_loss=cfg["agents"]["basal_energy_loss"])

	return interface, mutation_fn

#=======================================================================

def make_world(cfg, key):

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
	for k, v in data.items():
		if k=="alive" or not v.shape:
			continue
		if v.shape[0]==mask.shape[0]:
			data[k] = v[mask]
	data["reproduction_rates"] = data["offsprings"] / data["ages"]
	del data["alive"] 
	return data


#=======================================================================

class Simulator:
	#-------------------------------------------------------------------
	def __init__(self, 
				 world: GridWorld, 
				 key: jax.Array, 
				 log: bool=False,
				 ckpt_dir: str|None=None,
				 ckpt_freq: int=10_000):
		# ---
		self.world = world
		key_sim, key_aux = jr.split(key)
		self.key_sim = key_sim
		self.key_aux = key_aux
		self.log = log
		self.ckpt_dir = ckpt_dir
		self.ckpt_freq = ckpt_freq
		# ---

		def _log_clbk(data: dict):
			data = host_log_transform(data)
			wandb.log(data)
			return jnp.zeros((), dtype=bool)

		def _ckpt_clbck(state_dict):
			time = state_dict["env_state"].time
			filename = f"{ckpt_dir}/{int(time)}.pickle"
			with open(filename, "wb") as file:
				pickle.dump(state_dict, file)
			return jnp.zeros((),dtype=bool)

		if ckpt_dir:
			os.makedirs(ckpt_dir, exist_ok=True)


		def _simulation_step(state: EnvState, key: jax.Array)->EnvState:
			# ---
			key, key_step = jr.split(key)
			state, step_data = world.step(state, key_step)
			# ---
			if log:
				data = metrics_fn(state, step_data)
				io_callback(_log_clbk, jax.ShapeDtypeStruct((),bool), data)
			if ckpt_dir:
				_ = jax.lax.cond(
					jnp.mod(state.time, ckpt_freq)==0,
					_ckpt_clbck,
					lambda *a, **k: jnp.zeros((), dtype=bool),
					{"env_state": state, "key": key}
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
			world_state = self.world.init(key_init)
			print("initialization completed")

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
			cls(world, key, log=cfg["log"], ckpt_dir=cfg["ckpt_dir"], ckpt_freq=cfg["ckpt_freq"]), 
			cfg
		)

	#-------------------------------------------------------------------


#=======================================================================


def main():
	""""""
	parser = argparse.ArgumentParser()
	parser.add_argument("filename", type=str)
	args = parser.parse_args()
	simulator, cfg = Simulator.from_config_file(args.filename)

	if cfg["log"]: 
		wandb.init(project=cfg["project"], config=cfg)

	simulator.run_interactive()

	if cfg["log"]:
		wandb.finish()

	

if __name__ == '__main__':
	
	main()