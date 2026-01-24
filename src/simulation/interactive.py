from .simulation import Simulator, SimulationState
from .render import render

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

def run_interactive(simulator: Simulator, sim_state: SimulationState|None=None, *, key: jax.Array):

    key_sim, key_aux = jr.split(key)

    print(f"""
    Starting interactive simulation !

    Run name: {simulator.logger.name if simulator.logger else "no logging"}
    
    Commands:
    i, init, initialize: initialize simulation
    s, sim, simulate [steps]: simulate the world for a given number of steps (default: 1)
    r, render: render the world
    q: quit the simulation
    h, help: show this help message
    """)

    sim_state = sim_state

    while True:
        user_input = input("cmd: ")
        cmd, *args = user_input.strip().split(" ")
        
        # ---

        if cmd in ["s", "sim", "simulate"]:
            if sim_state is None:
                print("first initialize with init")
            else:
                if not args:
                    steps = 1
                else:
                    steps = int(args[0])
                print(f"Simulating {steps} steps...")
                key_sim, _key = jr.split(key_sim)
                sim_state, _ = simulator.rollout(sim_state, steps, key=_key)

        # ---

        elif cmd in ["init", "i", "initialize"]:
            if sim_state is not None:
                print("sim already initialized")
            else:
                key_sim, _key = jr.split(key_sim)
                sim_state = simulator.initialize(key=_key)

        # ---

        elif cmd in ["r", "render"]:
            if sim_state is None:
                print("first initialize with init")
            else:
                ax = plt.figure().add_subplot()
                render(simulator, sim_state, ax)
                plt.show()

        # ---

        elif cmd == "q":
            simulator.finish()
            return sim_state

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