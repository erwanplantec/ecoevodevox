import argparse
import jax
from jax import numpy as jnp, random as jr, nn as jnn
import wandb
import matplotlib.pyplot as plt
from src.simulation import Simulator, run_interactive


def main(args):
	""""""
	simulator, cfg = Simulator.from_config_file(args.filename)

	if simulator.logger is not None:
		simulator.logger.initialize(cfg)

	if args.debug:
		state = simulator.initialize(key=jr.key(1))
		state, trace = simulator.rollout(state, 16, key=jr.key(0))
		simulator.finish()

	elif args.interactive:

		run_interactive(simulator, key=jr.key(cfg["seed"]))

	else:
		keys = jr.split(jr.key(cfg["seed"]), args.repetitions)
		for key in keys:
			state = simulator.initialize(key=jr.key(1))
			state, trace = simulator.rollout(state, args.steps, key=jr.key(cfg["seed"]))
	

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("filename", type=str)
	parser.add_argument("--debug", action="store_true", help="simulate 16 steps")
	parser.add_argument("--interactive", action="store_true", help="run simulation in interactive mode")
	parser.add_argument("--steps", type=int, default=0)
	parser.add_argument("--repetitions", type=int, default=1)
	args = parser.parse_args()
	
	main(args)