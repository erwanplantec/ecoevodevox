
import argparse
import jax
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
		new_state, _ = world.step(state, jr.key(0))
		print(jax.tree.map(lambda a,b: a.dtype==b.dtype, state.agents_states, new_state.agents_states))

		plt.scatter(*state.agents_states.body.pos.T); plt.show()

	else:

		if cfg["log"]: 
			wandb.init(project=cfg["project"], config=cfg)

		simulator.run_interactive()

		if cfg["log"]:
			wandb.finish()

	

if __name__ == '__main__':
	
	main()