from .core import SimulationState

import jax
import numpy as np


def metrics_fn(sim_state: SimulationState, step_data: dict)->dict:
    """computes metrics from world state and step_data to log"""
    agents = sim_state.agents_states
    env = sim_state.env_state
    food_levels = {
        f"food_{i}": env.food[i].sum() for i in range(env.food.shape[0])
    }
    data = {
        "alive": agents.alive,
        "population": agents.alive.sum(),
        "energy_levels": agents.energy,
        "offsprings": agents.n_offsprings,
        **step_data,
        "ages": agents.age,
        **food_levels,
        "total_food": env.food.sum(),
        "network_sizes": agents.neural_state.mask.sum(-1),
        "body_sizes": agents.body.size
    }
    return data

def host_log_transform(data: dict)->dict:
    data = jax.tree.map(np.asarray, data)
    masked_data = {}
    mask = data["alive"]
    reduced_data = {}
    if not np.any(mask):
        return {"population": 0}
    for k, v in data.items():
        if k=="alive": continue
        if not v.shape:
            masked_data[k] = v
        elif v.shape[0]==mask.shape[0]:
            arr = v[mask]
            arr_max, arr_min, arr_avg, arr_var = np.max(arr), np.min(arr), np.mean(arr), np.var(arr)
            reduced_data[f"{k} (avg)"] = arr_avg
            reduced_data[f"{k} (max)"] = arr_max
            reduced_data[f"{k} (min)"] = arr_min
            reduced_data[f"{k} (var)"] = arr_var
            masked_data[k] = arr
            # if (~np.isnan(arr_min)) and (~np.isnan(arr_max)):
            #     data[k] = arr
    masked_data["reproduction_rates"] = masked_data["offsprings"] / masked_data["ages"]
    data = {**masked_data, **reduced_data}
    return data