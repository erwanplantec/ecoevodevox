# CLAUDE.md — EcoEvoDevox (EEDx)

## What this is

EcoEvoDevox (EEDx) is a **modular artificial-life platform for studying the ecological and
developmental roots of neural evolution**. It is the codebase behind:

> Plantec, Risi & Solé (2025). *Eco-Evo-Devox: A Modular Artificial Life Platform for Studying
> the Ecological and Developmental Roots of Neural Evolution.* ALIFE 2025 Proceedings, p.72.
> DOI 10.1162/ISAL.a.900.

A large population of embodied agents lives in a 2D `GridWorld`. Each agent has a **body**, a
**neural network** (possibly grown by a developmental encoding), and **sensory/motor interfaces**.
Agents move, sense diffusing **chemical** fields, **eat food**, spend **energy**, **reproduce**
(with mutation) when they have accumulated enough energy, and **die** from starvation or old age.
Evolution is **open-ended and non-episodic** — there is no explicit fitness function or generational
loop; selection emerges from survival and reproduction in the shared environment. The scientific
target is how development (ontogeny of neural circuits) and ecology (food, chemical niches) shape
neural evolution and **adaptive radiations**.

Everything is written in **JAX** and is heavily vectorized (`vmap`/`scan`/`lax.cond`) and
multi-device sharded, so tens of thousands of agents and large grids run on GPU/TPU.

## Tech stack

- **JAX** (`jax[cuda]` on Linux, plain `jax` on macOS) — all simulation math.
- **Equinox** (`eqx.Module`) — neural network / interface modules (parameters as pytree leaves).
- **flax.struct.PyTreeNode** — immutable state containers (`SimulationState`, `AgentState`, `EnvState`, configs). Update with `.replace(...)`.
- **wandb** (optional, `[logging]` extra) — metric logging.
- **matplotlib / celluloid** (optional, `[plotting]` extra) — rendering.
- Python **>=3.13**, dependency management via **uv** (`uv.lock`, `pyproject.toml`).

## Running

```bash
uv run python main.py <config.yml>              # headless rollout (--steps N, --repetitions R)
uv run python main.py <config.yml> --interactive # REPL: init / sim [n] / render / quit
uv run python main.py <config.yml> --debug       # run 16 steps as a smoke test
```

Programmatic entry point:

```python
from src.simulation import Simulator, run_interactive
simulator, cfg = Simulator.from_config_file("configs/nhnca.yml")
state = simulator.initialize(key=jr.key(0))
state, trace = simulator.rollout(state, steps=1000, key=jr.key(1))
```

`Simulator.from_config_file` (in `src/simulation/simulation.py`) is the wiring hub: it calls
`make_world`, `make_agents_interface` (both in `src/simulation/utils.py`) and builds the `Logger`.

## Config file format (IMPORTANT — this changed)

Config is plain YAML loaded with `yaml.safe_load` and interpreted in
**`src/simulation/utils.py`** (`load_config_file`, `make_world`, `make_agents_interface`) plus
`SimulationConfig(**cfg["simulation"])`. **`configs/nhnca.yml` is the canonical, up-to-date
example.** `configs/config.yml` and `configs/hypernet.yml` are **legacy** and will NOT load with
the current parser (old flat `env` block, obsolete `encoding:` block, missing required `agents`
keys). The top-level `README.md` also documents the *old* layout in places — trust the code and
`nhnca.yml`, not the README.

Top-level keys and how each is consumed:

| Key | Consumed by | Becomes |
| --- | --- | --- |
| `seed` | `main.py` | RNG seed |
| `simulation` | `SimulationConfig(**...)` | `max_agents`, `init_agents`, `birth_pool_size`, `wall_effect`, `wall_penalty` |
| `logging` | `Logger(...)` | `wandb_log`, `name`, `ckpt_freq`, `sampling_freq`, `sampling_size`, `wandb_project` |
| `ct-*` | `make_world` | one **ChemicalType** per key (see below) |
| `ft-*` | `make_world` | one **FoodType** per key |
| `env` | `GridworldConfig(**...)` | `size`, `walls_density`, `chemicals_detection_threshold`, `flow` |
| `agents` | `make_agents_interface` + `AgentConfig` | body/energy params + `nn`/`sensory`/`motor`/`mutation` sub-blocks |

**Chemical types** — every top-level key starting with `ct` defines one chemical, stacked in
declaration order (so `ct-1` is channel 0, `ct-2` is channel 1, …). Fields → `ChemicalType`:
`diffusion_rate`, `is_sparse`, `emission_rate`. The count of `ct-*` keys = number of chemical
channels agents can sense/emit.

**Food types** — every key starting with `ft` defines one food type. Fields → `FoodType`:
`growth_rate`, `dmin`, `dmax`, `chemical_signature`, `energy_concentration`,
`spontaneous_grow_prob`, `initial_density`. `growth_rate` may be a **number** (uniform) or a
**string expression** of the grid coordinates `x`, `y` for spatial heterogeneity, e.g.
`growth_rate: "0.005 * (1 - y)"`. Coordinates are normalised to `[0, 1)` (resolution-independent);
`x` is the first spatial axis (`Body.pos[0]`), `y` the second. Evaluated by `eval_growth_field`
(`src/eco/food.py`) into a per-cell `[H, W]` field stored as `FoodType.growth_field`; only `x`,
`y` and a small set of math funcs (`sin`, `cos`, `exp`, `sqrt`, `abs`, `tanh`, `where`, `clip`,
`pi`, …) are in scope. The world is toroidal, so a non-periodic expression like `1 - y` has a seam
at the wrap — use a periodic form (`0.5*(1+sin(2*pi*y))`) for a smooth gradient.

**Chemical signatures** — resolved by `resolve_chemical_signature` in `src/simulation/utils.py`
and accepted anywhere a signature is configured (`ft-*.chemical_signature`,
`agents.chemical_signature`). Three forms:

| Form | Example | Meaning |
| --- | --- | --- |
| **name** (preferred) | `chemical_signature: "ct-2"` | one-hot on that chemical; survives reordering/inserting `ct-*` keys |
| index | `chemical_signature: 1` | one-hot on channel 1 |
| list | `chemical_signature: [0.3, 0.7]` | verbatim blend; length must equal `#chemicals` |

Anything else (unknown name, out-of-range index, wrong-length list) raises at config-load time.

**`agents` block** — flat scalar keys are read explicitly into `AgentConfig`
(`max_age`, `init_energy`, `max_energy`, `basal_energy_loss`, `size_energy_cost`,
`min_body_size`, `max_body_size`, `body_resolution`, `time_below_threshold_to_die`,
`time_above_threshold_to_reproduce`, `reproduction_energy_cost`, and the optional
`chemical_signature` — what every agent emits, in the forms above; defaults to one-hot on the
first chemical. It is copied into every genotype at init and passed through mutation unchanged,
so it is constant over a run, not an evolving trait. It also contains four
**component sub-blocks**, each selected by a `which:` string and instantiated by name from a
registry; all other keys in the block are passed as constructor kwargs:

```yaml
agents:
  nn:       { which: "neuronca_ctrnn", size: 16, dev_steps: 50, ... }  # -> neural_models
  sensory:  { which: "spatially_embedded", sensor_expression_threshold: 0.1, body_resolution: 6 }  # -> sensory_interfaces
  motor:    { which: "braitenberg", interface: "se", max_wheel_speed: 8.0, ... }  # -> motor_interfaces
  mutation: { which: "generalized", sigma: 0.03, p_mut: 1.0, sigma_size: 0.1 }  # -> mutation_models
```

**The `which:` pattern is the core extensibility mechanism.** To add a component, implement the
class and register it in the relevant registry dict (see "Extension points" below).

## Repository layout

```
main.py                 CLI entry (headless / interactive / debug)
scripts/sim.py          alt runner
configs/                nhnca.yml (canonical), config.yml + hypernet.yml (LEGACY)
notebooks/demo.ipynb
src/
  simulation/           orchestration + I/O
    simulation.py       Simulator: initialize / step / rollout, birth+death logic, from_config_file
    core.py             SimulationConfig, SimulationState
    utils.py            load_config_file, make_world, make_agents_interface  <-- YAML parsing lives here
    metrics.py          metrics_fn (device) + host_log_transform (host)
    logging.py          Logger: wandb logging, checkpoints, agent sampling
    interactive.py      run_interactive REPL
    render.py           matplotlib rendering
    scenario.py
  eco/                  the environment ("Eco")
    gridworld.py        GridWorld + EnvState: food map, walls, chemical diffusion, vision, eating
    food.py             FoodType, FFT growth convolution
    chemicals.py        ChemicalType, FFT diffusion convolution (optional advective flow)
    mini.py, utils.py
  devo/                 the agent / body / brain ("Devo")
    interface.py        AgentInterface: encode -> neural_step -> decode, energy bookkeeping, body geometry
    core.py             AgentConfig, Genotype, Body, AgentState, Observation
    nn/                 neural + developmental encodings (registry: neural_models)
      ctrnn.py          CTRNN + IndirectCTRNN (base for grown networks)
      rnn.py            RNN
      rand.py           RAND_CTRNN (regulatory artificial neuro-development)
      hypernetwork.py   HyperRNN (convolutional hypernetwork decoder)
      hypernca.py       NeuronNCA ("neuronca_ctrnn"): NCA grows a spatially-embedded CTRNN
    sensory/           registry: sensory_interfaces
      spatially_embedded.py  neurons sense by their grown 2D position (epithelial/border neurons)
      flatten.py             flatten env obs into a vector (subset: all/edges/front)
      image.py
    motor/             registry: motor_interfaces
      braitenberg.py   2-wheel robot; interface "se" (spatial motor neurons) or "direct" (last 2 neurons)
      ciliated.py      (registered-out)
  evo/                 evolution ("Evo")
    core.py            MutationModel base + Genotype re-export
    mutation.py        GeneralizedMutation (per-parameter Gaussian mutation)
  utils/               viz.py, log.py
data/                  checkpoints & samples (zipped on finish)
wandb/                 wandb run logs
```

The `eco` / `devo` / `evo` split mirrors the paper's **Eco-Evo-Devo** framing.

## Core data model (`src/devo/core.py`, `src/simulation/core.py`, `src/eco/gridworld.py`)

- **`Genotype`** = `(neural_params, body_size, chemical_emission_signature)`. This is what mutates
  and is inherited. `neural_params` is the eqx-array pytree of the chosen NN model. (Currently the
  chemical emission signature is fixed to channel 0 at init and *not* mutated — flagged in code as a
  planned extension.)
- **`AgentState`** = genotype + `body` (`Body(pos, heading, size)`) + `neural_state` +
  `sensory_state` + `motor_state` + bookkeeping (`alive`, `age`, `energy`,
  `time_above_threshold`, `time_below_threshold`, `n_offsprings`, `generation`, `id_`, `parent_id_`).
- **`SimulationState`** = `(env_state, agents_states, time)`. Agents are stored in **fixed-size
  buffers of length `max_agents`**; dead/empty slots are padded. Birth reuses free (`~alive`) slots.
- **`EnvState`** = `(food, walls, time, last_agent_id)`. `food` is a boolean `[F, H, W]` map (one
  channel per food type); `walls` is `[H, W]` bool.

## Simulation step (`Simulator.step` / `step_agents`)

Per tick:
1. **Food update** — `world.update_food`: FFT growth convolution spreads food near existing sources
   (`dmin`/`dmax` annulus, `growth_rate`) plus `spontaneous_grow_prob`; walls/occupied cells blocked.
2. **Observations** — agent bodies are discretized to grid cells (`get_body_points`,
   `body_resolution × body_resolution` sample points). Agents **emit** chemicals per their signature;
   `compute_chemical_fields` sums food + agent chemical sources and applies the FFT diffusion kernel
   (sparse chemicals become Bernoulli events). Each agent samples the chemical + wall fields at its
   body points.
3. **Agent step** (`vmap` over `AgentInterface.step`): `encode_observation` (sensory) →
   `neural_step` (NN) → `decode_neural` (motor action). Energy is decremented by
   size + basal + motor + neural + sensory costs.
4. **Actions** — `apply_agents_actions` moves bodies (Braitenberg kinematics), wraps toroidally,
   applies `wall_effect` (`none` / `penalize` / `kill`).
5. **Eating** — `world.share_food_and_update`: food energy in each occupied cell is split among the
   agents' body parts there; eaten food cells are cleared.
6. **Death & reproduction** — `death_and_reproduction`:
   - **Death** if `age > max_age` OR `time_below_threshold > time_below_threshold_to_die`.
   - **Reproduction** if `time_above_threshold > time_above_threshold_to_reproduce`. Parents are
     top-k sampled into a `birth_pool_size` buffer (with noise for stochastic selection), children
     get **mutated** genotypes, are placed behind the parent, and inherit incremented `generation`.
     Guarded by `lax.cond` (only runs when someone reproduces and free slots exist).

`rollout` is a `jax.lax.scan` over `step`. Pass `with_trace=True` to accumulate per-step state/data.

## Developmental encodings (the "Devo" novelty)

Neural networks can be **directly encoded** (`ctrnn`, `rnn`: genotype = the weights) or **grown**
from a compact genotype:

- **`NeuronNCA` (`"neuronca_ctrnn"`, `nn/hypernca.py`)** — a Neural Cellular Automaton runs for
  `dev_steps` on a `size × size` grid of channels; the resulting per-cell channels are read out as
  neurons whose 2D position `x`, synapse channels, bias, time constant `tau`, and sensory/motor
  gene expression (`s`, `m`) define a spatially-embedded CTRNN. Wiring `W` is built from
  `nb_wiring_rules` bilinear rules over synapse channels. This is the encoding in `nhnca.yml`.
- **`RAND_CTRNN` (`"rand_ctrnn"`, `nn/rand.py`)** — regulation-based artificial neuro-development
  (cells divide/migrate/die via regulatory/migratory/signalling genes) producing a CTRNN.
- **`HyperRNN` (`"hyper_rnn"`, `nn/hypernetwork.py`)** — convolutional hypernetwork decoder.

Grown networks subclass **`IndirectCTRNN`** and produce a state carrying `x` (neuron positions),
`W`, `tau`, `gain`, `bias`, `mask`, and `s`/`m` (sensory/motor expression). The
**`spatially_embedded` sensory** and **`braitenberg` `se` motor** interfaces require these fields —
i.e. spatially-embedded sensing/acting only works with a developmental, spatially-embedded network.

## Extension points (how to add things)

All four agent components resolve `which:` against a registry dict — **register your class there**:

| Component | Registry | File |
| --- | --- | --- |
| Neural / dev encoding | `neural_models` | `src/devo/nn/__init__.py` |
| Sensory interface | `sensory_interfaces` | `src/devo/sensory/__init__.py` |
| Motor interface | `motor_interfaces` | `src/devo/motor/__init__.py` |
| Mutation model | `mutation_models` | `src/evo/__init__.py` |

- **New NN**: subclass `NeuralModel` (`nn/core.py`) with `__init__(..., *, key)`, `init(key) -> state`,
  and `__call__(x, state, key) -> (state, energy_cost)`. `make_apply_init` adapts the eqx module into
  the `(params, ...)` functional form the `AgentInterface` uses.
- **New sensory/motor**: subclass `SensoryInterface` / `MotorInterface`; implement `init` + `encode`
  (sensory) or `init` + `decode` + `move` (motor). Ensure compatibility with the NN's state fields.
- **New mutation**: subclass `MutationModel` (`evo/core.py`), implement `mutate_neural_params`;
  it receives `genotype_like` to know parameter structure.

The three agent components must be **mutually compatible** (the README stresses this): a
spatially-embedded sensory/motor interface needs a spatially-embedded (grown) network state.

## Logging & outputs (`src/simulation/logging.py`, `metrics.py`)

- `metrics_fn` runs **on-device** (population, energies, ages, offsprings, food levels, body sizes,
  step data); `host_log_transform` runs **host-side** to mask dead agents and compute
  avg/max/min/var, pushed to wandb via `io_callback`.
- `ckpt_freq` pickles full `SimulationState` to `data/<name>/ckpts/`; `sampling_freq` pickles random
  living-agent samples to `data/<name>/samples/`. On `finish()` the run dir is zipped.
- wandb project defaults to `eedx` (overridable via `logging.wandb_project`).

## Conventions & gotchas

- **State is immutable**: use `.replace(field=...)` on `PyTreeNode`s; never mutate in place.
- **Fixed-size buffers**: population lives in `max_agents`-length arrays; dead agents occupy padded
  slots. Metrics must mask by `agents.alive`. `max_agents % nb_devices == 0` is asserted (sharding).
- **Heavy `float16`**: bodies, energies, actions are `float16` for memory; watch precision/overflow.
- **Toroidal world**: positions wrap (`normalize_posture`); world dims **must be even** (FFT convs).
- **Non-episodic evolution**: no fitness function; don't look for one. Selection is implicit via the
  energy/reproduction/death loop.
- **README & legacy configs are partly stale** — the current source of truth for config schema is
  `src/simulation/utils.py` + `configs/nhnca.yml`.
- Everything is JIT/`vmap`/`scan`-traced — keep new code JAX-pure (no Python-side control flow on
  traced values; use `lax.cond`/`where`).
