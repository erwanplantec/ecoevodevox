# EcoEvoDevox (EEDx)

EcoEvoDevox is a **modular artificial-life platform for studying the ecological and developmental
roots of neural evolution**, written in **JAX** and heavily vectorised (`vmap`/`scan`/`lax.cond`)
so that tens of thousands of agents and large grids run on GPU/TPU.

A large population of embodied agents lives in a 2D toroidal `GridWorld`. Each agent has a **body**,
a **neural network** (possibly *grown* by a developmental encoding), and **sensory/motor**
interfaces. Agents move, sense diffusing **chemical** fields, **eat food**, spend **energy**,
**reproduce** (with mutation) once they have accumulated enough energy, and **die** from starvation
or old age. Evolution is **open-ended and non-episodic**: there is no fitness function or
generational loop — selection emerges from survival and reproduction in the shared environment.

> Plantec, Risi & Solé (2025). *Eco-Evo-Devox: A Modular Artificial Life Platform for Studying the
> Ecological and Developmental Roots of Neural Evolution.* ALIFE 2025 Proceedings, p.72.
> DOI 10.1162/ISAL.a.900.

```python
from src.simulation import Simulator

simulator, cfg = Simulator.from_config_file("configs/rand_baseline.yml")
state = simulator.initialize(key=jr.key(0))
state, trace = simulator.rollout(state, steps=1000, key=jr.key(1))
```

## Running

```bash
uv run python scripts/sim.py <config.yml> --steps N --repetitions R  # headless rollout
uv run python scripts/sim.py <config.yml> --interactive             # REPL: init / sim [n] / render / quit
uv run python scripts/sim.py <config.yml> --debug                   # run 16 steps as a smoke test
```

Or drive it programmatically (see the snippet above), or interactively in the browser via the web
app (`from src.simulation.webapp import launch; launch("configs/rand_baseline.yml")`).

`configs/rand_baseline.yml` is the **canonical, tuned, up-to-date example**. (`config.yml`,
`hypernet.yml` are legacy and will not load with the current parser.)

---

# Configuration

Config is plain YAML loaded with `yaml.safe_load` and interpreted in `src/simulation/utils.py`
(`load_config_file`, `make_world`, `make_agents_interface`) plus `SimulationConfig(**cfg["simulation"])`.

| Top-level key | Consumed by | Becomes |
| --- | --- | --- |
| `seed` | `main.py` | RNG seed |
| `simulation` | `SimulationConfig(**...)` | population/buffer params |
| `logging` | `Logger(...)` | wandb logging, checkpoints, sampling |
| `ct-*` | `make_world` | one **chemical type** per key (declaration order = channel order) |
| `ft-*` | `make_world` | one **food type** per key |
| `env` | `GridworldConfig(**...)` | grid geometry |
| `agents` | `make_agents_interface` + `AgentConfig` | body/energy params + component sub-blocks |

## `simulation`

```yaml
simulation:
  max_agents: 10000       # fixed-size population buffer; must be divisible by nb_devices
  init_agents: 2048       # founders at t=0
  birth_pool_size: 64     # max births per step (<= max_agents)
  wall_effect: none       # "none" | "penalize" | "kill"
  wall_penalty: 1.0       # energy lost per wall contact when wall_effect == "penalize"
```

## `logging`

```yaml
logging:
  wandb_log: true
  name: my_run            # run name (optional)
  wandb_project: eedx
  ckpt_freq: null         # pickle full SimulationState every N steps (null = off)
  sampling_freq: 10000    # pickle a random living-agent sample every N steps
  sampling_size: 32
```

## `env`

```yaml
env:
  size: [1024, 1024]                 # grid dims (must be even — FFT convolutions)
  walls_density: 0.0                 # Bernoulli wall probability per cell
  chemicals_detection_threshold: 1.e-5
  flow: null                         # optional [dx, dy] advective flow on chemical diffusion
```

The world is **toroidal**: positions wrap.

## Chemical types (`ct-*`)

Every top-level key starting with `ct` defines one chemical channel, stacked in **declaration
order** (so `ct-1` is channel 0, `ct-2` channel 1, …). The number of `ct-*` keys = the number of
channels agents can sense and emit into.

```yaml
ct-1:
  diffusion_rate: 5.0     # width of the FFT diffusion kernel
  is_sparse: false        # true -> field is sampled as Bernoulli events
  emission_rate: 1.0
```

## Food types (`ft-*`)

Every key starting with `ft` defines one food type.

```yaml
ft-1:
  growth_rate: 0.005            # number, OR an expression of x, y (see below)
  dmin: 1                       # inner radius of the growth annulus (cells)
  dmax: 1                       # outer radius
  chemical_signature: 1         # what this food emits — name / index / list (see below)
  energy_concentration: 2       # energy per eaten cell
  spontaneous_grow_prob: 1e-8   # per-cell chance of appearing from nowhere
  initial_density: 0.0          # Bernoulli fill at t=0
```

**Spatial growth rates.** `growth_rate` may be a **number** (uniform) or a **string expression** of
the grid coordinates `x`, `y`, giving food that grows at different rates in different places:

```yaml
ft-1: { growth_rate: "0.0025 * (1 - cos(2*pi*y))", ... }   # fertile band across the middle
```

- Coordinates are normalised to `[0, 1)` (resolution-independent). `x` is the first spatial axis
  (`Body.pos[0]`), `y` the second — the axes an agent moves along.
- In scope: `x`, `y` and `sin cos tan exp log sqrt abs tanh sign minimum maximum clip where pi`.
- The world is **toroidal**, so a non-periodic expression like `1 - y` has a seam at the wrap; use
  a periodic form (`0.5*(1+sin(2*pi*y))`, `1-cos(2*pi*y)`) for a smooth gradient.
- Negative values are clipped to 0 (with a warning). Evaluated by `eval_growth_field`
  (`src/eco/food.py`) into a per-cell `FoodType.growth_field`.

## Chemical signatures (`ft-*.chemical_signature`, `agents.chemical_signature`)

Resolved by `resolve_chemical_signature` (`src/simulation/utils.py`); accepted in three forms:

| Form | Example | Meaning |
| --- | --- | --- |
| **name** (preferred) | `"ct-2"` | one-hot on that chemical; survives reordering `ct-*` keys |
| index | `1` | one-hot on channel 1 |
| list | `[0.3, 0.7]` | verbatim blend; length must equal the number of chemicals |

Bad input (unknown name, out-of-range index, wrong length) raises at config-load time.

## `agents`

Flat scalar keys go straight into `AgentConfig`; four sub-blocks configure the agent's components.

```yaml
agents:
  # --- body / energy ---
  max_age: 10000
  init_energy: 1.5
  max_energy: 15.0
  basal_energy_loss: 0.02
  size_energy_cost: 0.0
  min_body_size: 4.0
  max_body_size: 4.0
  body_resolution: 8                       # body sampled on a body_resolution^2 grid of points
  time_above_threshold_to_reproduce: 100
  time_below_threshold_to_die: 50
  reproduction_energy_cost: 0.0
  eat_energy_fraction: 1.0                 # only eat below this fraction of max_energy (satiation)
  chemical_signature: "ct-2"               # what agents emit; default one-hot on channel 0

  # --- components (each selected by `which:`; other keys -> constructor kwargs) ---
  nn:       { which: rand_ctrnn, ... }
  sensory:  { which: spatially_embedded, ... }
  motor:    { which: ciliated_torque, ... }
  mutation: { which: generalized, ... }
```

`eat_energy_fraction` < 1 makes satiation real (a nearly-full agent walks over food without
eating it, so food can accumulate into patches instead of being grazed flat); 1.0 is the classic
"eat unless completely full" behaviour.

The three interface components must be **mutually compatible**: the `spatially_embedded` sensory
interface and the `ciliated` / `ciliated_torque` / `braitenberg "se"` motor interfaces require a
spatially-embedded (grown) network state.

### Neural / developmental encodings — `agents.nn`

Registry `neural_models` (`src/devo/nn/__init__.py`).

| `which` | Description | Code |
| --- | --- | --- |
| `ctrnn` | Continuous-time RNN (direct: genotype = weights) | [ctrnn.py](./src/devo/nn/ctrnn.py) |
| `rnn` | Plain RNN (direct) | [rnn.py](./src/devo/nn/rnn.py) |
| `rand_ctrnn` | **RAND** — regulation-based artificial neuro-development grows a CTRNN | [rand.py](./src/devo/nn/rand.py) |
| `neuronca_ctrnn` | **NeuronNCA** — an NCA grows a spatially-embedded CTRNN | [hypernca.py](./src/devo/nn/hypernca.py) |
| `hyper_rnn` | Convolutional hypernetwork decoder | [hypernetwork.py](./src/devo/nn/hypernetwork.py) |

Grown encodings (`rand_ctrnn`, `neuronca_ctrnn`) produce a spatially-embedded network carrying
neuron positions `x`, wiring `W`, `tau`/`gain`/`bias`, `mask`, and sensory/motor expression `s`/`m`
— the fields the spatially-embedded sensory and ciliated/braitenberg-`se` motor interfaces need.

### Sensory interfaces — `agents.sensory`

Registry `sensory_interfaces` (`src/devo/sensory/__init__.py`).

| `which` | Description | Code |
| --- | --- | --- |
| `spatially_embedded` | Neurons sense by their grown 2D position (border/epithelial neurons) | [spatially_embedded.py](./src/devo/sensory/spatially_embedded.py) |
| `flatten` | Flatten the env observation into a vector (subset: all/edges/front) | [flatten.py](./src/devo/sensory/flatten.py) |
| `retina` | Neuron layout used as a retina over an image input | [retina.py](./src/devo/sensory/retina.py) |

> Contract for `spatially_embedded`: `sensory_genes == n_chemicals + 1 (walls) + 4 (internal signals)`.

### Motor interfaces — `agents.motor`

Registry `motor_interfaces` (`src/devo/motor/__init__.py`).

| `which` | Description | Code |
| --- | --- | --- |
| `braitenberg` | Two-wheel robot; `interface: "se"` (spatial motor neurons) or `"direct"` (last 2 neurons) | [braitenberg.py](./src/devo/motor/braitenberg.py) |
| `ciliated` | Border cilia split into velocity vs turning by dominant axis; action `[velocity, omega]` | [ciliated.py](./src/devo/motor/ciliated.py) |
| `ciliated_torque` | Every cilium both propels and steers (rigid-body torque); holonomic, action `[vx, vy, omega]` | [ciliated_torque.py](./src/devo/motor/ciliated_torque.py) |

For the ciliated interfaces the motor gene `m` is read as **cilium size** (larger cilium → more
thrust), continuous with no expression threshold.

### Mutation models — `agents.mutation`

Registry `mutation_models` (`src/evo/__init__.py`).

| `which` | Description | Code |
| --- | --- | --- |
| `generalized` | Per-parameter Gaussian mutation (`sigma`, `p_mut`, `sigma_size`) | [mutation.py](./src/evo/mutation.py) |

`Genotype` = `(neural_params, body_size, chemical_emission_signature)`. Body size mutates; the
chemical emission signature is fixed at init (see `agents.chemical_signature`).

---

# Extension points

Add a component by implementing the class and registering it in the relevant registry dict:

| Component | Registry | File |
| --- | --- | --- |
| Neural / dev encoding | `neural_models` | `src/devo/nn/__init__.py` |
| Sensory interface | `sensory_interfaces` | `src/devo/sensory/__init__.py` |
| Motor interface | `motor_interfaces` | `src/devo/motor/__init__.py` |
| Mutation model | `mutation_models` | `src/evo/__init__.py` |

---

# Tooling

- **Interactive web app** (`src/simulation/webapp.py`, `launch("configs/rand_baseline.yml")`) —
  build/run/watch simulations in the browser: live world render (colour agents by energy, speed,
  age, or neuron count), streaming metric charts, a chemical-field overlay, a food-painting brush,
  an agent inspector (grown network + live internals, tracked across the map), a spawn-agent tool,
  adjustable/full speed, checkpoints, and a MiniEnv replay to probe a clicked agent in isolation.

- **Mini environments** (`src/eco/mini.py`) — single-agent arenas for probing an interface in
  isolation: `MiniTaxis` (chemotaxis to one beacon) and `MiniMultiTaxis` (sequential beacons,
  fitness = beacons reached plus proximity shaping). Used for unit tests and QD scoring.

- **Quality-Diversity / MAP-Elites** (`src/evo/qd.py`, `src/evo/qd_rand.py`) — MAP-Elites over RAND
  genotypes with swappable fitness, descriptors and behaviour stats; scoring functions for taxis,
  MNIST classification, developmental plasticity, and embodiment.

- **Rendering & animation** (`src/simulation/render.py`, `src/utils/viz.py`) — fast RGB frame
  renderer, developmental-trajectory scatter, network plot, MAP-Elites repertoire plot, and
  `animate_mini_rollout` / `animate_multitaxis_rollout` (the latter optionally plays the network's
  development before the rollout).

- **Checkpointing** (`src/simulation/checkpoint.py`) — portable save/load of a full
  `SimulationState`.
