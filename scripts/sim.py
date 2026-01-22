import argparse
import jax
from jax import numpy as jnp, random as jr, nn as jnn
import wandb
import matplotlib.pyplot as plt
from src.simulation import Simulator, run_interactive


def main():
	""""""
	parser = argparse.ArgumentParser()
	parser.add_argument("filename", type=str)
	parser.add_argument("--debug", type=int, default=0)
	args = parser.parse_args()
	simulator, cfg = Simulator.from_config_file(args.filename)

	if args.debug:
		state = simulator.initialize(key=jr.key(1))
		state, trace = simulator.rollout(state, 16, key=jr.key(0))
		simulator.finish()

	else:

		if simulator.logger is not None:
			simulator.logger.initialize(cfg)

		run_interactive(simulator, key=jr.key(cfg["seed"]))

	

if __name__ == '__main__':
	
	main()