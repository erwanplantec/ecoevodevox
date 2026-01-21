from .simulation import Simulator, SimulationState

import jax
import jax.numpy as jnp
import jax.random as jr

def run_interactive(simulator: Simulator, sim_state: SimulationState):

    key_sim, key_aux = jr.split(sim_state.key)

    print(f"""
    Starting interactive simulation !

    Run name: {simulator.name}
    
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
            print(f"Simulating {steps} steps...")
            key_sim, _key = jr.split(key_sim)
            world_state = simulator.simulate(sim_state, _key, steps)

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