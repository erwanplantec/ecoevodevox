from ..eco.gridworld import EnvState

import jax
import numpy as np


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