class ScenarioSimulator():
    #-------------------------------------------------------------------
    def __init__(
        self, 
        meta_cfg_file: str, 
        log: bool=False,
        ckpt_dir: str|None=None,
        ckpt_freq: int=10_000, 
        n_devices: int|None=None):


        cfg = load_config(meta_cfg_file)
        simulators = []
        key = jr.key(cfg["seed"])

        
        for k, value in cfg.items():

            if k.startswith("env"):
                key, k1, k2 = jr.split(key, 3)
                cfg_file = value["file"]
                cfg = load_config(cfg_file)
                wrld = make_world(cfg, k1)
                sim = Simulator(wrld, k2, log, ckpt_dir, ckpt_freq, n_devices)

                _init_food = jax.jit(wrld._init_food, out_shardings=None) #type:ignore

                def _step(state: EnvState, key: jax.Array)->EnvState:
                    if value["on_start"]=="init":
                        key, _key = jr.split(key)
                        state = jax.lax.cond(
                            state.time==value["from"],
                            lambda s, k: s.replace(food=s.food|_init_food(k)),
                            lambda s, k: s,
                            state, _key
                        )
                    state = sim.simulation_step(state, key)
                    return state

                simulators.append((_step, value))

        
        key, key_init = jr.split(key)
        for x in simulators:
            print(x[1])
        simulators = sorted(simulators, key=lambda x: x[1]["from"])
        state = wrld.init(key_init) #type:ignore
        state = state.replace(food=jnp.zeros_like(state.food))
        self.state = state

        def _simulation_step(state: EnvState, key: jax.Array)->EnvState:
            
            done = jnp.asarray(False)
            for sim_fn, cfg in simulators:


                state, done = jax.lax.cond(
                    (cfg["from"]<=state.time) & (state.time<cfg["to"]) & (~done),
                    lambda s, k: (sim_fn(s,k), jnp.asarray(True)),
                    lambda s, k: (s, done),
                    state, key
                )

            return state

        self.simulation_step = _simulation_step


        @partial(jax.jit, static_argnames=("steps"))
        def _simulate(state: EnvState, key: jax.Array, steps: int)->EnvState:

            def _step(_, c):
                state, key = c
                key, _key = jr.split(key)
                state = _simulation_step(state, _key)
                return state, key

            state, _ = jax.lax.fori_loop(
                0, steps, _step, (state, key)
            )

            return state

        self.simulate = _simulate