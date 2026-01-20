import jax
import jax.numpy as jnp
import equinox as eqx

from ..agents import (sensory_interfaces, SensoryInterface, 
                      motor_interfaces, MotorInterface, 
                      AgentInterface, make_apply_init, 
                      Policy, nn_models)
from ..evo import MutationModel, mutation_models, Genotype
from ..devo import encoding_models

def make_agents_interface(cfg: dict, key: jax.Array)->tuple[AgentInterface, MutationModel]:
    """initializes the agents interface and the mutation function"""
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
    policy: Policy = policy_fctry(key)
    policy_prms, _ = eqx.partition(policy, eqx.is_array)
    policy_apply, policy_init = make_apply_init(policy, reshape_prms=False)
    # ---
    mut_cfg = cfg["agents"]["mutation"]
    cls = mutation_models.get(mut_cfg["which"], None); assert cls is not None, f"mutation mdl {mut_cfg['which']} is not valid"
    kwargs = {k:v for k,v in mut_cfg.items() if k !="which"}
    genotype_like = Genotype(policy_prms, jnp.asarray(0.0))
    mutation_fn: MutationModel = cls(genotype_like=genotype_like, **kwargs)
    # ---

    interface = AgentInterface(policy_apply=policy_apply,
                               policy_init=policy_init,
                               policy_fctry=prms_fctry,
                               sensory_interface=sensory_interface,
                               motor_interface=motor_interface,
                               body_resolution=cfg["agents"]["body_resolution"],
                               basal_energy_loss=cfg["agents"]["basal_energy_loss"])

    return interface, mutation_fn