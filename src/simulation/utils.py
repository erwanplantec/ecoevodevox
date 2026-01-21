import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jr
import equinox as eqx
import yaml

from ..agents import (sensory_interfaces, SensoryInterface, 
                      motor_interfaces, MotorInterface, 
                      AgentInterface, make_apply_init, 
                      Policy, nn_models, AgentConfig)
from ..evo import MutationModel, mutation_models, Genotype
from ..eco.gridworld import EnvState, FoodType, ChemicalType, GridWorld, GridworldConfig
from ..devo import encoding_models


def load_config_file(filename)->dict:
    with open(filename, "r") as file:
        cfg = yaml.safe_load(file)
    return cfg


def make_agents_interface(cfg: dict)->tuple[AgentInterface, MutationModel]:
    """initializes the agents interface and the mutation function"""
    assert "agents" in cfg.keys()
    #---
    motor_cfg = cfg["agents"]["motor"]
    motor_cls = motor_interfaces.get(motor_cfg["which"],None); assert motor_cls is not None, f"motor interface {motor_cfg['which']} is not valid"
    motor_kwargs = {k:v for k,v in motor_cfg.items() if k !="which"}
    motor_interface: MotorInterface = motor_cls(**motor_kwargs)
    #---
    sensory_cfg = cfg["agents"]["sensory"]
    sensory_cls = sensory_interfaces.get(sensory_cfg["which"], None); assert sensory_cls is not None, f"sensory interface {sensory_cfg['which']} is not valid"
    sensory_kwargs = {k:v for k,v in sensory_cfg.items() if k !="which"}
    sensory_interface: SensoryInterface = sensory_cls(**sensory_kwargs)
    #---
    enc_cfg = cfg["agents"]["encoding"]
    enc_cls = encoding_models.get(enc_cfg['which'], None); assert enc_cls is not None, f"encoding model {enc_cfg['which']} is not valid"
    enc_kwargs = {k:v for k,v in enc_cfg.items() if k !="which"}
    encoding_fctry = lambda key: enc_cls(**enc_kwargs, key=key)
    #---
    nn_cfg = cfg["agents"]["nn"]
    nn_cls = nn_models.get(nn_cfg["which"], None); assert nn_cls is not None, f"nn model {nn_cfg['which']} is not valid"
    nn_kwargs = {k:v for k,v in nn_cfg.items() if k !="which"}
    policy_fctry = lambda key: nn_cls(encoding_model=encoding_fctry(key), **nn_kwargs)
    prms_fctry = lambda key: eqx.filter(policy_fctry(key), eqx.is_array)
    policy: Policy = policy_fctry(jr.key(0))
    policy_prms, _ = eqx.partition(policy, eqx.is_array)
    policy_apply, policy_init = make_apply_init(policy, reshape_prms=False)
    # ---
    mut_cfg = cfg["agents"]["mutation"]
    cls = mutation_models.get(mut_cfg["which"], None); assert cls is not None, f"mutation mdl {mut_cfg['which']} is not valid"
    kwargs = {k:v for k,v in mut_cfg.items() if k !="which"}
    nb_ct = len([k for k in cfg.keys() if k.startswith("ct")])
    genotype_like = Genotype(policy_prms, jnp.asarray(0.0), jnp.zeros(nb_ct))
    mutation_fn: MutationModel = cls(genotype_like=genotype_like, **kwargs)
    # ---

    agent_cfg = AgentConfig(max_age=cfg["agents"]["max_age"],
                            init_energy=cfg["agents"]["init_energy"],
                            max_energy=cfg["agents"]["max_energy"],
                            basal_energy_loss=cfg["agents"]["basal_energy_loss"],
                            size_energy_cost=cfg["agents"]["size_energy_cost"],
                            min_body_size=cfg["agents"]["min_body_size"],
                            max_body_size=cfg["agents"]["max_body_size"],
                            body_resolution=cfg["agents"].get("body_resolution", None),
                            time_below_threshold_to_die=cfg["agents"]["time_below_threshold_to_die"],
                            time_above_threshold_to_reproduce=cfg["agents"]["time_above_threshold_to_reproduce"],
                            reproduction_energy_cost=cfg["agents"]["reproduction_energy_cost"])

    agents_interface = AgentInterface(cfg=agent_cfg, 
                                      policy_apply=policy_apply,
                                      policy_init=policy_init,
                                      policy_fctry=prms_fctry,
                                      sensory_interface=sensory_interface,
                                      motor_interface=motor_interface)

    return agents_interface, mutation_fn


def make_world(cfg: dict)->tuple[GridWorld, GridworldConfig]:
    """initializes the world"""

    env_cfg = cfg["env"]

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
                      chemical_types=chemical_types, 
                      food_types=food_types)

    return world, world_cfg