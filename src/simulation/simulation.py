from jaxtyping import PyTree
from matplotlib.pyplot import stem
from .metrics import host_log_transform, metrics_fn
from ..eco.gridworld import GridWorld, get_cell_index
from ..devo import AgentInterface, Body, Action
from ..evo import MutationModel, Genotype
from .core import SimulationState
from .core import SimulationConfig, SimulationState, AgentState
from .logging import Logger
from .utils import make_world, make_agents_interface, load_config_file
from ..settings import POSITION_DTYPE

import jax
from jax import numpy as jnp, random as jr
from jax.sharding import PartitionSpec as P, NamedSharding
import pickle

#=======================================================================


class Simulator:

    #-------------------------------------------------------------------

    def __init__(self, 
                 cfg: SimulationConfig,
                 world: GridWorld,  # The gridworld environment
                 agent_interface: AgentInterface, # The agents interface,
                 mutation_fn: MutationModel, # the mutation function,
                 # ---
                 nb_devices: int|None=None, # Number of devices to use (None for all available)
                 # ---
                 logger: Logger|None=None
                ):  
        # ---
        self.cfg = cfg
        self.world = world
        self.agent_interface = agent_interface
        self.mutation_fn = mutation_fn
        self.logger = logger
        # ---
        self.nb_devices = jax.device_count() if nb_devices is None else nb_devices
        assert self.cfg.max_agents % self.nb_devices == 0
        # ---

        # jax>=0.11 make_mesh defaults to Explicit axis types, but with_sharding_constraint
        # (used in initialize/step) requires Auto axes -> request them explicitly
        device_mesh = jax.make_mesh((self.nb_devices,), ('N',), axis_types=(jax.sharding.AxisType.Auto,))
        self.state_shardings = SimulationState(
            agents_states=NamedSharding(device_mesh, P("N")), #type:ignore
            env_state=NamedSharding(device_mesh, P()), #type:ignore
            time=NamedSharding(device_mesh, P()) #type:ignore
        )

    # ------------------------------------------------------------------

    def initialize(self, *, key: jax.Array) -> SimulationState:

        key_agents, key_world = jr.split(key)

        agents_states = self.initialize_agents(key=key_agents)

        env_state = self.world.init(key=key_world)

        sim_state = SimulationState(env_state, agents_states, jnp.zeros((), dtype=jnp.uint32))
        
        return jax.lax.with_sharding_constraint(sim_state, self.state_shardings)

    # ------------------------------------------------------------------

    def initialize_agents(self, *, key: jax.Array)->AgentState:
        def _pad(x):
            pad_values = [(0,self.cfg.max_agents-self.cfg.init_agents)] + [(0,0)]*(x.ndim-1)
            return jnp.pad(x, pad_values)

        key_prms, key_pos, key_head, key_size, key_init, key_age = jr.split(key, 6)
        neural_params = jax.vmap(self.agent_interface.neural_fctry)(jr.split(key_prms,self.cfg.init_agents))
        body_sizes = jr.uniform(key_size, (self.cfg.init_agents,), minval=self.agent_interface.cfg.min_body_size, maxval=self.agent_interface.cfg.max_body_size, dtype=jnp.float16) 
        # What agents emit, from `agents.chemical_signature` in the config; defaults to one-hot on
        # the first chemical, which is what the setup did before the key existed. Identical for
        # every agent and passed through mutation unchanged, so it is constant over a run (making
        # it an evolvable trait is the planned extension).
        sig = self.agent_interface.cfg.chemical_signature
        if sig is None:
            agents_chemical_signature = jnp.zeros((self.cfg.init_agents, self.world.nb_chemicals)).at[:, 0].set(1.0)
        else:
            sig = jnp.asarray(sig, dtype=jnp.float32)
            assert sig.shape == (self.world.nb_chemicals,), (
                f"agents.chemical_signature has {sig.shape[0]} entries but the world has "
                f"{self.world.nb_chemicals} chemical(s)")
            agents_chemical_signature = jnp.broadcast_to(sig, (self.cfg.init_agents, self.world.nb_chemicals))
        genotypes = Genotype(neural_params, body_sizes, agents_chemical_signature)
        positions = jr.uniform(key_pos, (self.cfg.init_agents, 2), minval=1.0, maxval=jnp.array(self.world.cfg.size, dtype=POSITION_DTYPE)-1, dtype=POSITION_DTYPE)
        headings = jr.uniform(key_head, (self.cfg.init_agents,), minval=0.0, maxval=2*jnp.pi, dtype=POSITION_DTYPE)
        ids_ = jnp.arange(1, self.cfg.init_agents+1, dtype=jnp.uint32)
        agents_states = jax.vmap(self.agent_interface.init)(genotypes,
                                                            positions,
                                                            headings,
                                                            ids_,
                                                            key=jr.split(key_init, self.cfg.init_agents))

        # spread initial ages over [1, 0.2*max_age] instead of starting every agent at age 1:
        # a synchronized founding cohort reproduces in one pulse and dies of old age all at
        # once, imposing an artificial max_age-periodic boom/bust on the population. Keeping
        # the spread well below max_age leaves the founders most of their lifespan to establish
        max_init_age = max(int(0.2 * self.agent_interface.cfg.max_age), 1)
        ages = jr.randint(key_age, (self.cfg.init_agents,), 1, max_init_age + 1).astype(jnp.uint16)
        agents_states = agents_states.replace(age=ages)

        agents_states = jax.tree.map(_pad, agents_states)

        return agents_states

    #-------------------------------------------------------------------

    def step(self, sim_state: SimulationState, *, key: jax.Array) -> tuple[SimulationState, dict]:

        key_food, key_agents, key_logger = jr.split(key, 3)
        
        # --- 1. update food ---
        env_state = self.world.update_food(sim_state.env_state, key_food)
        sim_state = sim_state.replace(env_state=env_state)

        # --- 2. agents step ---

        sim_state, agents_step_data = self.step_agents(sim_state, key=key_agents)

        # --- 3. update state ---

        step_data = {**agents_step_data}
        sim_state = sim_state.replace(time=sim_state.time+1)

        # --- 7. do logging stuff ---
        if self.logger is not None:
            self.logger.log(sim_state, step_data, key_logger)

        return jax.lax.with_sharding_constraint(sim_state, self.state_shardings), step_data


    #-------------------------------------------------------------------

    def agents_chemical_sources(self, agents_states, bodies_points: jax.Array)->jax.Array:
        """Chemical emitted by the agents themselves, splatted onto the grid -> [C, H, W].

        Each agent deposits its `chemical_emission_signature` at every one of its body points, so
        a bigger body emits proportionally more. Factored out of `step_agents` so a viewer can
        reproduce the *exact* field the agents sense rather than an approximation of it.
        """
        bodies_points_cells = get_cell_index(bodies_points) # N, 2, R, R
        res = bodies_points.shape[-1]
        agents_emissions = jnp.tile(agents_states.genotype.chemical_emission_signature[...,None,None],
                                    (1, 1, res, res))
        # dead slots keep their genotype, so gate emission on `alive` or corpses would keep
        # scenting the map from wherever they happened to die
        agents_emissions = jnp.where(agents_states.alive[:,None,None,None], agents_emissions, 0.0)
        return (
            jnp.zeros((self.world.nb_chemicals, *self.world.cfg.size))
            .at[:, *bodies_points_cells.transpose(1,0,2,3).reshape(2,-1)]
            .add(agents_emissions.transpose(1,0,2,3).reshape(self.world.nb_chemicals,-1))
        )

    #-------------------------------------------------------------------

    def chemical_fields(self, sim_state: SimulationState, *, key: jax.Array)->jax.Array:
        """The diffused chemical fields as the agents perceive them -> [C, H, W].

        Same path as `step_agents`: food signatures plus agent emissions, run through the
        diffusion convolution, sparse channels sampled and sub-threshold values zeroed. Exposed
        for visualisation and analysis (the step itself does not return the field).
        """
        bodies_points = jax.vmap(self.get_body_points)(sim_state.agents_states.body)
        sources = self.agents_chemical_sources(sim_state.agents_states, bodies_points)
        return self.world.compute_chemical_fields(sim_state.env_state, sources, key=key)

    #-------------------------------------------------------------------

    def step_agents(self, sim_state: SimulationState, *, key: jax.Array)->tuple[SimulationState, dict]:

        key_obs, key_agents, key_death_and_repr = jr.split(key, 3)

        # --- 2. retrieve world infos for agents ---
        bodies_points = jax.vmap(self.get_body_points)(sim_state.agents_states.body) # N, 2, R, R
        agents_chemical_sources = self.agents_chemical_sources(sim_state.agents_states, bodies_points)

        env_obs = self.world.get_agents_observations(sim_state.env_state, bodies_points, agents_chemical_sources, key=key_obs)

        # --- 3. update agents internal states and retrieve action ---
        actions, agents_states, agents_step_data = jax.vmap(self.agent_interface.step)(
            env_obs, sim_state.agents_states, jr.split(key_agents, self.cfg.max_agents)
        )
        sim_state = sim_state.replace(agents_states=agents_states)

        # --- 4. apply effect of actions ---
        sim_state = self.apply_agents_actions(actions, sim_state)

        # --- 5. let agents eat ---
        sim_state = self.update_agents_and_food(sim_state)

        # --- 6. manage deaths and reproductions
        sim_state, death_and_reproduction_data = self.death_and_reproduction(sim_state, key_death_and_repr)

        return sim_state, {**agents_step_data, **death_and_reproduction_data}

    # ------------------------------------------------------------------

    def apply_agents_actions(self, actions: Action, sim_state: SimulationState) -> SimulationState:
        """move agents according to actions and apply effects of env (walls)"""
        
        agents_states, env_state = sim_state.agents_states, sim_state.env_state
        # --- 1. move agents and check wall contact ---
        moved_bodies = jax.vmap(self.agent_interface.move)(actions, sim_state.agents_states.body)
        new_bodies = self.world.normalize_posture(moved_bodies)
        # measure the step from the pre-wrap position: normalize_posture wraps toroidally, so
        # crossing a boundary would otherwise count as a ~world-width jump
        step_distance = jnp.linalg.norm((moved_bodies.pos - agents_states.body.pos).astype(jnp.float32), axis=-1)
        agents_distance = agents_states.distance_travelled + jnp.where(agents_states.alive, step_distance, 0.0)
        new_bodies_points = jax.vmap(self.agent_interface.get_body_points)(new_bodies)
        makes_contact = jax.vmap(self.world.check_wall_contact, in_axes=(None,0))(env_state, new_bodies_points) * sim_state.agents_states.alive
        # --- 2. apply effect of contact ---
        if self.cfg.wall_effect=="kill":
            agents_alive = agents_states.alive & (~makes_contact)
            agents_energy = agents_states.energy
        elif self.cfg.wall_effect=="penalize":
            agents_alive = agents_states.alive
            agents_energy = jnp.where(makes_contact&agents_states.alive, agents_states.energy-self.cfg.wall_penalty, agents_states.energy)
        elif self.cfg.wall_effect=="none":
            agents_alive = agents_states.alive
            agents_energy = agents_states.energy
        else:
            raise ValueError(f"wall effect {self.cfg.wall_effect} is not valid")

        new_agents_states = agents_states.replace(body=new_bodies, alive=agents_alive, energy=agents_energy,
                                                  distance_travelled=agents_distance)

        return sim_state.replace(agents_states=new_agents_states)

    #-------------------------------------------------------------------

    def update_agents_and_food(self, sim_state) -> SimulationState:
        """let agents eatif possible, update their energy accordingly and update food"""
        agents_states, env_state = sim_state.agents_states, sim_state.env_state

        eating_agents = self.agent_interface.is_eating(agents_states)

        bodies_points = jax.vmap(self.get_body_points)(agents_states.body)

        agents_food_energy_intake, env_state = self.world.share_food_and_update(env_state, bodies_points, eating_agents)

        agents_states = self.agent_interface.update_energy(agents_states, agents_food_energy_intake)

        return sim_state.replace(agents_states=agents_states, env_state=env_state)

    #-------------------------------------------------------------------

    def death_and_reproduction(self, sim_state: SimulationState, key: jax.Array) -> tuple[SimulationState, dict]:
        """"""
        key_repro, key_mut = jr.split(key, 2)

        agents_states = sim_state.agents_states
        
        below_threshold = agents_states.energy < 0.0
        above_threshold = ~below_threshold
        
        agents_tat = jnp.where(above_threshold&agents_states.alive, agents_states.time_above_threshold+1, 0); assert isinstance(agents_tat, jax.Array)
        agents_tbt = jnp.where(below_threshold&agents_states.alive, agents_states.time_below_threshold+1, 0); assert isinstance(agents_tbt, jax.Array)

        # --- 1. Death ---

        dead = self.agent_interface.is_dying(agents_states)
        dead = dead & agents_states.alive
        avg_dead_age = jnp.where(dead, agents_states.age, 0).sum() / dead.sum() #type:ignore
        agents_alive = agents_states.alive & ( ~dead )
        agents_age = jnp.where(agents_alive, agents_states.age, 0)
        agents_states = agents_states.replace(alive=agents_alive, time_above_threshold=agents_tat, time_below_threshold=agents_tbt, age=agents_age)

        # --- 2. Reproduce ---

        def _reproduce(reproducing: jax.Array, agents_states: AgentState, key: jax.Array)->AgentState:
            """
            """
            key_shuff, key_head, key_init = jr.split(key, 3)

            free_buffer_spots = ~agents_states.alive # N,
            _, parents_buffer_id = jax.lax.top_k(reproducing+jr.uniform(key_shuff,reproducing.shape,minval=-0.1,maxval=0.1), self.cfg.birth_pool_size) # add random noise to have non deterministic sammpling
            # top_k always returns birth_pool_size indices, so it pads with agents that are NOT
            # reproducing (and may be dead). Keep the original mask to filter them out: rebuilding
            # `reproducing` from parents_buffer_id would make parents_mask all-True and let every
            # padded slot spawn a child at its (stale) position.
            parents_mask = reproducing[parents_buffer_id]
            parents_genotypes = jax.tree.map(lambda x: x[parents_buffer_id], agents_states.genotype)
            is_free, childs_buffer_id = jax.lax.top_k(free_buffer_spots, self.cfg.birth_pool_size)
            childs_mask = parents_mask & is_free #is a child if parent was actually reproducing and there are free buffer spots

            # agents that actually produced a child: only these pay the reproduction cost
            reproducing = jnp.zeros(self.cfg.max_agents, dtype=jnp.bool).at[parents_buffer_id].set(childs_mask)

            childs_buffer_id = jnp.where(childs_mask, childs_buffer_id, self.cfg.max_agents) # assign dummy index if not born
            parents_buffer_id = jnp.where(childs_mask, parents_buffer_id, self.cfg.max_agents) # assign dummy index if not parent
            
            childs_genotypes = jax.vmap(self.mutation_fn)(parents_genotypes, jr.split(key_mut, self.cfg.birth_pool_size))

            parents_bodies = jax.tree.map(lambda x: x[parents_buffer_id], agents_states.body)
            direction = jnp.mod(parents_bodies.heading + jnp.pi, 2*jnp.pi)
            delta = jnp.stack([jnp.cos(direction), jnp.sin(direction)], axis=-1)
            childs_positions = agents_states.body.pos[parents_buffer_id] + delta*(parents_bodies.size*2+1.0)[:,None] 
            childs_headings = jr.uniform(key_head, minval=0.0, maxval=2*jnp.pi, dtype=POSITION_DTYPE, shape=(self.cfg.birth_pool_size,))

            childs_ids = jnp.where(childs_mask, jnp.cumsum(childs_mask, dtype=jnp.uint32)+agents_states.id_.max()+1, 0)
            childs_parents_ids = agents_states.id_[parents_buffer_id]

            childs_generation = agents_states.generation[parents_buffer_id]+1

            childs_states = jax.vmap(self.agent_interface.init)(genotype=childs_genotypes, 
                                                                position=childs_positions, 
                                                                heading=childs_headings, 
                                                                id_=childs_ids, 
                                                                parent_id_=childs_parents_ids, 
                                                                generation=childs_generation,
                                                                key=jr.split(key_init, self.cfg.birth_pool_size))

            agents_states = jax.tree.map(lambda s, c: s.at[childs_buffer_id].set(c), agents_states, childs_states)
            
            agents_states = self.agent_interface.update_after_reproduction(agents_states, reproducing)

            return agents_states
        # ---   

        reproducing = self.agent_interface.is_reproducing(agents_states)

        agents_states = jax.lax.cond(
            jnp.any(reproducing)&jnp.any(~agents_states.alive),
            _reproduce, 
            lambda repr, agts, key: agts, 
            reproducing, agents_states, key_repro
        )

        sim_state = sim_state.replace(
            agents_states=agents_states
        )

        return (
            sim_state, 
            dict(reproducing=reproducing, dying=dead, avg_dead_age=avg_dead_age)
        )

    #-------------------------------------------------------------------

    def get_body_points(self, body: Body):
        return self.agent_interface.get_body_points(body)

    # ------------------------------------------------------------------

    def rollout(self, sim_state: SimulationState, steps: int, with_trace: bool=False, *, key: jax.Array) -> tuple[SimulationState, PyTree|None]:
        """Run up to `steps` steps, stopping early once the population is extinct.

        Without a trace this uses a `while_loop`, so a run that dies out costs nothing for its
        remaining steps. `sim_state.time` tells you how many steps actually ran. With
        `with_trace=True` the scan is kept (the stacked trace needs a static length), so the
        full `steps` are always executed.
        """
        keys = jr.split(key, steps)

        if with_trace:
            def _step(sim_state, key):
                new_sim_state, data = self.step(sim_state, key=key)
                return new_sim_state, dict(sim_state=sim_state, step_data=data)
            return jax.lax.scan(_step, sim_state, keys)

        def _cond(carry):
            i, sim_state = carry
            return (i < steps) & sim_state.agents_states.alive.any()

        def _body(carry):
            i, sim_state = carry
            # index the pre-split keys so the RNG stream matches the scan version
            new_sim_state, _ = self.step(sim_state, key=keys[i])
            return i + 1, new_sim_state

        _, sim_state = jax.lax.while_loop(_cond, _body, (0, sim_state))
        return sim_state, None

    # ------------------------------------------------------------------

    def finish(self):
        if self.logger is not None:
            self.logger.finish()

    #-------------------------------------------------------------------
    
    def load_ckpt(self, filename: str) -> SimulationState:
        """Load a checkpoint and return the SimulationState (the simulator is stateless)."""
        from .checkpoint import load_state
        sim_state, _meta = load_state(filename)
        return sim_state

    def save_ckpt(self, filename: str, sim_state: SimulationState, meta: dict | None=None) -> str:
        """Write `sim_state` to `filename`."""
        from .checkpoint import save_state
        return save_state(filename, sim_state, meta)

    # ==================================================================

    @classmethod
    def from_config_file(cls, filename: str):
        """create a simulatore from a config file"""
        cfg = load_config_file(filename)
        # --- 1. create world --- 
        world, world_cfg = make_world(cfg)
        # --- 2. creat agent interface ---
        agent_interface, mutation_fn = make_agents_interface(cfg)
        # --- 3. create logger ---
        log_cfg = cfg["logging"]
        logger = Logger(wandb_log=log_cfg.get("wandb_log", False),
                        name=log_cfg.get("name", None),
                        ckpt_freq=log_cfg.get("ckpt_freq", None),
                        sampling_freq=log_cfg.get("sampling_freq", None),
                        sampling_size=log_cfg.get("sampling_size", None),
                        metrics_fn=metrics_fn,
                        host_log_transform=host_log_transform,
                        wandb_project=log_cfg.get("wandb_project", "eedx"))
        logger.initialize(cfg)
        # ---
        sim_cfg = SimulationConfig(**cfg["simulation"])
        simulator = Simulator(cfg=sim_cfg, 
                              world=world,
                              agent_interface=agent_interface,
                              mutation_fn=mutation_fn,
                              nb_devices=cfg.get("nb_devices", None),
                              logger=logger)
        return simulator, cfg



if __name__ == '__main__':
    sim, cfg = Simulator.from_config_file("configs/dev_cfg.yml")
    sim_state = sim.initialize(key=jr.key(0))
    sim_state = sim.step(sim_state, key=jr.key(1))







