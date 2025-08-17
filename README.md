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
| Name | Description |
| --- | ----------- |
| `spatially_embedded` | ... |
| `flatten` | ... |

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
| Name | Description |
| --- | ----------- |
| `ctrnn` | ... |
| `rnn` | ... |
| `ffnn` | ... |

Example:
```yaml
nn:
    which: "ctrnn"  
```

### Neural Network Encoding

Encoding models define how the parameters of the neural network are encoded and are called at the intialization of the policy state. 

YAML definition:
```yaml
encoding: 
    which: "name_of_encoding_model"
    constructor_arg: value
```

| Name | Description |
| --- | ----------- |
| `rand` | ... |
| `direct_rnn` | ... |
| `direct_ctrnn` | ... |

Example:

```yaml
encoding:
    which: "rand"
    nb_neurons: 1
    max_neurons: 128
    regulatory_genes: 4
    migratory_genes: 4
    signalling_genes: 4
    sensory_genes: 5
    motor_genes: 1
    synaptic_genes: 4
    nb_synaptic_rules: 4
    synaptic_proteins: 4
    signalling_proteins: 4
    max_mitosis: 10
    mitotic_factor_threshold: 10.0
    apoptosis_factor_threshold: 10.0
    autonomous_decay: true
    dev_iters: 400
    network_type: "ctrnn"
```

### Motor Interface

YAML definition:
```yaml
motor:
    which: "name_of_motor_model"
    constructor_arg: value
```
Where `"name_of_motor_model"` must be referenced in `src.agents.motor.motor_interfaces`

| Name | Description |
| --- | ----------- |
| `braitenberg` | ... |
| `ciliated` | ... |

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