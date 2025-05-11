import jax
from jax import numpy as jnp, random as jr, nn as jnn
import equinox as eqx
import equinox.nn as nn
import evosax as ex
import wandb
from jax.experimental import io_callback

from src.eco.mini import make as make_env
from src.agents.motor import motor_interfaces, MotorInterface
from src.agents.sensory import sensory_interfaces, SensoryInterface
from src.devo import encoding_models, DevelopmentalModel
from src.agents.nn import nn_models, Policy, make_apply_init
from src.evo import mutation_models, MutationModel
from src.agents.core import Genotype
from src.agents.interface import AgentInterface


def make_agent(cfg: dict, key: jax.Array):
	motor_cfg = cfg["motor"]
	cls = motor_interfaces.get(motor_cfg["which"],None); assert cls is not None, f"motor interface {motor_cfg['which']} is not valid"
	kwargs = {k:v for k,v in motor_cfg.items() if k !="which"}
	motor_interface: MotorInterface = cls(**kwargs)

	#---
	

	sensory_cfg = cfg["sensory"]
	cls = sensory_interfaces.get(sensory_cfg["which"], None); assert cls is not None, f"sensory interface {sensory_cfg['which']} is not valid"
	kwargs = {k:v for k,v in sensory_cfg.items() if k !="which"}
	sensory_interface: SensoryInterface = cls(**kwargs)

	# ---

	enc_cfg = cfg["encoding"]
	cls = encoding_models.get(enc_cfg['which'], None); assert cls is not None, f"encoding model {enc_cfg['which']} is not valid"
	kwargs = {k:v for k,v in enc_cfg.items() if k !="which"}
	encoding_model: DevelopmentalModel = cls(**kwargs, key=key)

	# ---

	nn_cfg = cfg["nn"]
	cls = nn_models.get(nn_cfg["which"], None); assert cls is not None, f"nn model {nn_cfg['which']} is not valid"
	kwargs = {k:v for k,v in nn_cfg.items() if k !="which"}
	policy: Policy = cls(encoding_model=encoding_model, **kwargs)
	policy_prms, _ = eqx.partition(policy, eqx.is_array)
	policy_apply, policy_init = make_apply_init(policy, reshape_prms=False)

	# ---

	def policy_fctry(key: jax.Array):
		return policy_prms

	# ---

	interface = AgentInterface(policy_apply=policy_apply,
							   policy_init=policy_init,
							   policy_fctry=policy_fctry,
							   sensory_interface=sensory_interface,
							   motor_interface=motor_interface,
							   body_resolution=cfg["body_resolution"],
							   basal_energy_loss=cfg["basal_energy_loss"])

	return interface, policy_prms


def make_train_fn(cfg, key):

	agent_interface, prms = make_agent(cfg["agent"], key=key)
	env = make_env(cfg["env"], agent_interface)
	eval_reps = cfg["algo"].get("eval_reps", 1)

	@jax.vmap
	def evaluate(prms, key):
		fitness, _ = jax.vmap(env.evaluate, in_axes=(None,0))(prms, jr.split(key, eval_reps))
		return fitness.mean()

	Strategy = getattr(ex, cfg["algo"]["strategy"])
	es: ex.Strategy = Strategy(popsize=cfg["algo"]["popsize"], pholder_params=prms, n_devices=1, **cfg["algo"]["args"])

	def _log_clbck(data):
		wandb.log(data)
		return jnp.zeros((), dtype=bool)

	def train_fn(key):

		if cfg["log"]:
			wandb.init(project=cfg["project"], config=cfg)

		def train_step(es_state, key):
			key_ask, key_eval = jr.split(key)
			prms, es_state = es.ask(key_ask, es_state)
			fitness = evaluate(prms, jr.split(key_eval, cfg["algo"]["popsize"]))
			es_state = es.tell(prms, -fitness, es_state)

			if cfg["log"]:
				data = dict(max_fitness=fitness.max(), avg_fitness=fitness.mean(), std_fitness=jnp.var(fitness), min_fitness=jnp.min(fitness))
				_ = io_callback(_log_clbck, jnp.zeros((), dtype=bool), data)

			return es_state, fitness


		key_init, key_evo = jr.split(key)
		es_state = es.initialize(key_init)
		es_state, fitneses = jax.block_until_ready(jax.lax.scan(train_step, es_state, jr.split(key_evo, cfg["generations"])))
		if cfg["log"]:
			wandb.finish()
		return es_state, fitneses

	return train_fn




if __name__ == '__main__':
	import yaml
	import argparse
	import matplotlib.pyplot as plt

	parser = argparse.ArgumentParser()
	parser.add_argument("filename")
	args = parser.parse_args()
	with open(args.filename, "r") as file:
		cfg = yaml.safe_load(file)

	seed = jr.key(cfg["seed"])
	key_make, key_train = jr.split(seed)
	train_fn = make_train_fn(cfg, key_make)
	es_state, fitness = train_fn(key_train)

	plt.plot(fitness.max(-1))
	plt.show()