# EcoEvoDevox (EEDx)


```python
from src.simulation import Simulator

simulator, cfg = Simulator.from_config_file("my_cfg_file.yml")
simulator.run_interactive()
```

<details>
<summary>Defining Agents</summary>

## Agents

The agent interface is defined by three main components:
- the sensory interface transforming raw observations from environment into input to the neural network
- the neural network updating internal state based on input from sensory interface
- the motor interface outputing an action based on the state of neural network

(the user should make sure that each of these component are compatible together)


### Sensory Interface


YAML definition:
```yaml
sensory:
    which: "name_of_sensory_model"
    constructor_arg: value
    ...
```
| Name | Description | Code |
| --- | ----------- | --- |
| `spatially_embedded` | Sensory interface for spatially embedded neural networks. Neurons inputs depend on their position in the environment.  | [spatially_embedded.py](./src/devo/sensory/spatially_embedded.py) | 
| `flatten` | Transforms env and internal input into flat observation vector. Can take a structured subset of observation.  | [flatten.py](./src/devo/sensory/flatten.py)

Where the name of sensory model must be referenced in `src.agents.sensory.sensory_interfaces`
Example:
```yaml
sensory:
    which: "spatially_embedded"
    sensor_expression_threshold: 0.9
    border_threshold: 0.9
```

### Neural Network


YAML definition:
```yaml
nn:
    which: "name_of_nn_model"
    constructor_arg: value
    ...
```

| Name | Description | code |
| --- | ---------- | --- |
| `ctrnn` | Continuous time reccurrent network | [ctrnn.py](./src/devo/nn/ctrnn.py) |
| `rnn` | Regular recurrent neural network without gating | [rnn.py](./src/devo/nn/rnn.py) |
| `hyper_rnn` | Convolutional decoder encoding  | [hypernet.py](./src/devo/nn/hypernet.py) |
| `rand_ctrnn` | Regulation based Artificial NeuroDevlopment | [rand.py](./src/devo/nn/rand.py) |

Example:
```yaml
nn:
    which: "ctrnn" 
    nb_neurons: 256 
```

### Motor Interface

YAML definition:
```yaml
motor:
    which: "name_of_motor_model"
    constructor_arg: value
```
Where `"name_of_motor_model"` must be referenced in `src.agents.motor.motor_interfaces`

| Name | Description | Code |
| --- | ----------- | --- |
| `braitenberg` |  Braitenberg like motor apparatus (2 wheeled robot) | [braitenbder.py](./src/devo/motor/braitenberg.py) |

</details>

<details>
<summary>Defining Environment</summary>

## Environment

```python
world = GridWorld()
```

YAML definition:
```yaml
env:
  size: [128,128]
  max_agents: 2048
  init_agents: 64
  birth_pool_size: 128
  max_age: 2000
  reproduction_cost: 0.5
  max_energy: 50.0
  initial_energy: 1.0
  time_above_threshold_to_reproduce: 80
  time_below_threshold_to_die: 50
  chemicals_detection_threshold: 0.001
  walls_density: 0.0
  wall_effect: "none"
```

### Food Types

```yaml
ft-x:

```

### Chemical Types


```yaml
ct-x:
    
```

</details>