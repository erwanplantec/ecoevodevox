
import argparse
from jax import numpy as jnp, random as jr, nn as jnn
import wandb
import matplotlib.pyplot as plt

from src.simulation import Simulator




def main():
	""""""
	parser = argparse.ArgumentParser()
	parser.add_argument("filename", type=str)
	parser.add_argument("--debug", type=int, default=0)
	args = parser.parse_args()
	simulator, cfg = Simulator.from_config_file(args.filename)

	if args.debug:
		world = simulator.world
		state = world.init(jr.key(1))
		state, _ = world.step(state, jr.key(0))
		state, _ = world.step(state, jr.key(1))
		plt.scatter(*state.agents_states.position.pos.T); plt.show()

	else:

		if cfg["log"]: 
			wandb.init(project=cfg["project"], config=cfg)

		simulator.run_interactive()

		if cfg["log"]:
			wandb.finish()

	

if __name__ == '__main__':
	
	main()