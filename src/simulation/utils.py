import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jr
import equinox as eqx
import yaml

from ..devo import (sensory_interfaces, SensoryInterface, 
                    motor_interfaces, MotorInterface, 
                    AgentInterface, make_apply_init, NeuralModel,
                    neural_models, AgentConfig)
from ..evo import MutationModel, mutation_models, Genotype
from ..eco.gridworld import EnvState, FoodType, ChemicalType, GridWorld, GridworldConfig
from ..eco.food import eval_growth_field


def load_config_file(filename)->dict:
    with open(filename, "r") as file:
        cfg = yaml.safe_load(file)
    return cfg


def chemical_names(cfg: dict)->list[str]:
    """The `ct-*` keys in declaration order — which is the channel order `make_world` stacks them
    in, so index i in any signature vector is `chemical_names(cfg)[i]`."""
    return [k for k in cfg.keys() if k.startswith("ct")]


def resolve_chemical_signature(value, names: list[str], what: str)->jnp.ndarray:
    """Turn a config-level chemical signature into a `[n_chemicals]` vector.

    Accepts, in order of preference:
      * a **name**, e.g. ``"ct-1"`` -> one-hot on that chemical. Preferred: it survives
        reordering or inserting a chemical type, whereas a bare index silently repoints.
      * an **int** index -> one-hot on that channel.
      * a **list/tuple** of length ``n_chemicals`` -> used verbatim, so a source can emit a blend.

    `what` names the thing being configured, for error messages.
    """
    n = len(names)
    if isinstance(value, str):
        if value not in names:
            raise ValueError(f"{what}: unknown chemical {value!r}; declared chemicals are {names}")
        return jnn.one_hot(names.index(value), n)
    # bool is an int subclass, and `True` as a channel index is a config typo, not channel 1
    if isinstance(value, int) and not isinstance(value, bool):
        if not 0 <= value < n:
            raise ValueError(f"{what}: chemical index {value} out of range for {n} chemical(s) {names}")
        return jnn.one_hot(value, n)
    if isinstance(value, (list, tuple)):
        if len(value) != n:
            raise ValueError(f"{what}: chemical signature has {len(value)} entries but {n} "
                             f"chemical(s) are declared {names}")
        return jnp.asarray(value, dtype=jnp.float32)
    raise TypeError(f"{what}: chemical signature must be a chemical name, an index or a "
                    f"length-{n} list, got {value!r}")


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
    nn_cfg = cfg["agents"]["nn"]
    nn_cls = neural_models.get(nn_cfg["which"], None); assert nn_cls is not None, f"nn model {nn_cfg['which']} is not valid"
    nn_kwargs = {k:v for k,v in nn_cfg.items() if k !="which"}
    neural_fctry = lambda key: nn_cls(**nn_kwargs, key=key)
    neural_prms_fctry = lambda key: eqx.filter(neural_fctry(key), eqx.is_array)
    # ---
    mut_cfg = cfg["agents"]["mutation"]
    cls = mutation_models.get(mut_cfg["which"], None); assert cls is not None, f"mutation mdl {mut_cfg['which']} is not valid"
    kwargs = {k:v for k,v in mut_cfg.items() if k !="which"}
    names = chemical_names(cfg)
    nb_ct = len(names)
    genotype_like = Genotype(neural_prms_fctry(jr.key(0)), jnp.asarray(0.0), jnp.zeros(nb_ct))
    mutation_fn: MutationModel = cls(genotype_like=genotype_like, **kwargs)
    # ---

    # optional: what agents emit. Left None (-> one-hot on the first chemical) when unset, which
    # is what every config did before this key existed.
    sig = cfg["agents"].get("chemical_signature", None)
    sig = None if sig is None else tuple(
        float(v) for v in resolve_chemical_signature(sig, names, "agents.chemical_signature"))

    # satiation threshold as a fraction of max_energy; 1.0 keeps the historical "eat unless full"
    eat_frac = float(cfg["agents"].get("eat_energy_fraction", 1.0))
    if not 0.0 < eat_frac <= 1.0:
        raise ValueError(f"agents.eat_energy_fraction must be in (0, 1], got {eat_frac}")

    agent_cfg = AgentConfig(chemical_signature=sig,
                            eat_energy_fraction=eat_frac,
                            max_age=cfg["agents"]["max_age"],
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
                                      neural_model_constructor=neural_fctry,
                                      sensory_interface=sensory_interface,
                                      motor_interface=motor_interface)

    return agents_interface, mutation_fn


def make_world(cfg: dict)->tuple[GridWorld, GridworldConfig]:
    """initializes the world"""

    env_cfg = cfg["env"]
    world_cfg = GridworldConfig(**env_cfg)
    size = tuple(int(s) for s in world_cfg.size)

    cfg_ct = {k:v for k,v in cfg.items() if k.startswith("ct")}
    cfg_ft = {k:v for k,v in cfg.items() if k.startswith("ft")}

    chemical_types = jax.tree.map(
        lambda *ct: jnp.stack(ct), *[ct for ct in cfg_ct.values()]
    )
    chemical_types = ChemicalType(**chemical_types)

    names = list(cfg_ct.keys())
    for typ in cfg_ft.keys():
        cfg_ft[typ]["chemical_signature"] = resolve_chemical_signature(
            cfg_ft[typ]["chemical_signature"], names, f"{typ}.chemical_signature")
        # `growth_rate` (a number or an expression of x, y) becomes a per-cell [H, W] growth field
        cfg_ft[typ]["growth_field"] = eval_growth_field(cfg_ft[typ].pop("growth_rate"), size)

    food_types = jax.tree.map(
        lambda *fts: jnp.stack([jnp.asarray(ft, dtype=jnp.float32) for ft in fts]),
        *list(cfg_ft.values()),
        is_leaf=lambda x: isinstance(x, list)
    )
    food_types = FoodType(**food_types)

    world = GridWorld(world_cfg,
                      chemical_types=chemical_types,
                      food_types=food_types)

    return world, world_cfg