from typing import Callable, NamedTuple
import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import evosax as ex
import equinox as eqx
from jaxtyping import PyTree
import realax as rx
from functools import partial
import wandb
import matplotlib.pyplot as plt
import numpy as np

from src.devo.policy_network.ctrnn import CTRNN
from src.eco.gridworld import (Agent, GridWorld,
							   EnvState,
							   FoodType,
							   ChemicalType,
							   Observation)
from src.devo.model_e import (Model_E, 
							  CTRNNPolicyConfig, 
							  mutate)
from src.devo.utils import make_apply_init


class Config(NamedTuple):
	seed: 	int=0
	debug: 	bool=False
	# --- log
	wandb_log: 		bool=True
	wandb_project: 	str="eedx"
	log_table_freq: int=1_000
	# --- world
	size:          		tuple[int,int] = (512,512)
	max_agents:      	int            = 10_000
	birth_pool_size:	int            = 2048
	cast_to_f16:    	bool           = False
	# ---
	wall_density:	float = 1e-3
	deadly_walls:	bool  = True 
	# --- food
	n_food_types:         	int   = 1
	growth_rate:  			float = 1e-2
	min_growth_dist: 		float = 1.0
	max_growth_dist: 		float = 1.0
	spontaneous_grow_prob: 	float = 1e-6
	initial_food_density:  	float = 1e-3
	energy_concentration:  	float = 1.0
	diffusion_rate:        	float = 20.0
	# --- agents
	initial_agents: 					int   = 256
	max_energy: 						float = 20.0
	base_energy_loss: 					float = 0.1
	reproduction_cost: 					float = 0.5
	applied_force_energy_cost: 			float = 0.0
	s_expression_energy_cost: 			float = 0.0
	m_expression_energy_cost: 			float = 0.0
	neurons_energy_cost: 				float = 0.0
	connection_cost:					float = 0.0
	time_below_threshold_to_die: 		int   = 30
	time_above_threshold_to_reproduce: 	int   = 50
	max_age: 							int   = 200
	# --- sensor interface
	fov: 							int   = 1
	sensor_expression_threshold: 	float = 0.03
	border_threshold: 				float = 0.9
	# --- motor interface
	motor_expression_threshold: float = 0.03
	force_threshold_to_move: 	float = 0.1
	neurons_max_motor_force: 	float = 0.1
	passive_eating: 			bool  = True
	passive_reproduction: 		bool  = True
	# --- dev model
	mdl: 					str   = "e"
	max_neurons: 			int   = 128
	n_synaptic_markers: 	int   = 4
	max_types: 				int   = 8
	T_dev: 					float = 10.
	dt_dev: 				float = 0.1
	temp_decay: 			float = 0.90
	extra_migration_fields: int   = 3
	N_gain: 				float = 50.
	connection_model: 		str   = "xoxt"
	# --- ctrnn prms
	T_ctrnn: 	float = 0.5
	dt_ctrnn: 	float = 0.1
	# --- mutations
	sigma_mut: 				float = 0.03
	p_mut: 					float = 1e-3
	p_duplicate_split: 		float = 1e-3
	p_duplicate_no_split: 	float = 1e-3
	p_rm: 					float = 1e-3
	p_add: 					float = 1e-3


def sensory_expression(
	ctrnn: CTRNN,
	sensor_activation: Callable=jnn.tanh,
	expression_threshold: float=0.03):
	# --- 
	assert ctrnn.s is not None
	assert ctrnn.mask is not None
	# ---
	s = ctrnn.s * ctrnn.mask[:,None]
	s = jnp.where(jnp.abs(s)>expression_threshold, s, 0.0)
	s = sensor_activation(s) #neurons,ds
	return s

def gridworld_sensory_interface(
	obs: Observation, 
	ctrnn: CTRNN, 
	fov: int=1,
	sensor_activation: Callable=jnn.tanh,
	expression_threshold: float=0.03,
	border_threshold: float=0.9):
	# ---
	assert ctrnn.s is not None
	# ---
	xs = ctrnn.x
	s = sensory_expression(ctrnn, sensor_activation, expression_threshold)
	# ---
	C = obs.chemicals # mC,W,W
	W = obs.walls
	mC, *_ = C.shape

	xs = xs * (1/(2*border_threshold)) #correct suche that rounding threshold is at border threshold
	coords = fov+jnp.round(xs.at[:,1].multiply(-1.) * fov).astype(jnp.int16)
	j, i = coords.T

	Ic = jnp.sum(C[:,i,j].T * s[:,:mC], axis=1) # chemical input #type:ignore
	Iw = jnp.sum(W[:,i,j].T * s[:,mC:mC+1], axis=1) # walls input #type:ignore
	Ii = jnp.sum(s[:, mC+1:] * obs.internal, axis=1) # internal input #type:ignore

	I = Ic + Iw + Ii
	return I

def motor_expression(
	ctrnn: CTRNN,
	border_threshold: float=0.9,
	expression_threshold:float=0.03,
	m_activation: Callable=lambda m: jnp.clip(m, 0.0, 1.0)):
	# ---
	assert ctrnn.m is not None
	assert ctrnn.mask is not None
	# ---
	m = ctrnn.m * ctrnn.mask[:,None]
	m = jnp.where(jnp.abs(m)>expression_threshold, m, 0.0)
	m = m_activation(m)
	on_border = jnp.any(jnp.abs(ctrnn.x)>border_threshold, axis=-1)
	m = jnp.where(on_border[:,None], m, 0.0)
	assert isinstance(m, jax.Array)
	return m


def gridworld_motor_interface(
	ctrnn: CTRNN, 
	border_threshold: float=0.9,
	expression_threshold:float=0.03,
	m_activation: Callable=lambda m: jnp.clip(m, 0.0, 1.0),
	threshold_to_move: float=0.1,
	neurons_max_force: float=0.1,
	pos_dtype: type=jnp.int16):
	# ---
	assert ctrnn.m is not None
	# --- 
	xs = ctrnn.x
	m = motor_expression(ctrnn, border_threshold, 
		expression_threshold, m_activation)[:,0]
	v = ctrnn.v
	# ---
	xs = xs * jnn.one_hot(jnp.argmax(jnp.abs(xs), axis=-1), 2)

	on_top = xs[:,1] 	> border_threshold
	on_bottom = xs[:,1] < - border_threshold
	on_right = xs[:,0] 	> border_threshold
	on_left = xs[:,0] 	< - border_threshold

	forces = jnp.clip(v*m, 0.0, neurons_max_force) #forces applied by all neurons (N,)
	N_force = jnp.where(on_bottom, 	forces, 0.0).sum() 	#type:ignore
	S_force = jnp.where(on_top, 	forces, 0.0).sum()	#type:ignore
	E_force = jnp.where(on_left, 	forces, 0.0).sum() 	#type:ignore
	W_force = jnp.where(on_right, 	forces, 0.0).sum() 	#type:ignore

	directional_forces = jnp.array([N_force, S_force, E_force, W_force]) # 4,
	directions = jnp.array([[-1,0],[1,0],[0,1],[0,-1]], dtype=pos_dtype)
	
	net_directional_force = jnp.sum(directional_forces[:,None] * directions, axis=0) #2,
	move = jnp.where(jnp.abs(net_directional_force)>threshold_to_move, #if force on component is above threshold
					 jnp.sign(net_directional_force), # move on unit
					 0.0).astype(jnp.int16) # don't move
	return move

#-------------------------------------------------------------------

def make_agents_model(cfg: Config):
	"""
	"""
	interface = CTRNNPolicyConfig(
		encode_fn=partial(gridworld_sensory_interface, 
						  fov=cfg.fov, 
						  border_threshold=cfg.border_threshold,
						  expression_threshold=cfg.sensor_expression_threshold),
		decode_fn=partial(gridworld_motor_interface, 
						  threshold_to_move=cfg.force_threshold_to_move,
						  border_threshold=cfg.border_threshold, 
						  neurons_max_force=cfg.neurons_max_motor_force),
		T=cfg.T_ctrnn, 

		dt=cfg.dt_ctrnn
	)

	sensory_dimensions = cfg.n_food_types + 3
	motor_dimensions = 1 + int(not cfg.passive_eating) + int(not cfg.passive_reproduction)

	if cfg.mdl=="e":
		def _fctry(key):
			model = Model_E(n_types=cfg.max_types, 
						    n_synaptic_markers=cfg.n_synaptic_markers,
						    max_nodes=cfg.max_neurons,
						    sensory_dimensions=sensory_dimensions,
						    motor_dimensions=motor_dimensions,
						    dt=cfg.dt_dev,
						    dvpt_time=cfg.T_dev,
						    temperature_decay=cfg.temp_decay,
						    extra_migration_fields=cfg.extra_migration_fields,
						    N_gain=cfg.N_gain,
						    policy_cfg=interface,
						    body_shape="square",
						    key=key)
			model = eqx.tree_at(lambda p: [p.types.pi,p.types.s, p.types.m], 
								model, 
								[model.types.pi.at[0].set(8.0/cfg.N_gain),
								 model.types.s.at[0,:].set(cfg.sensor_expression_threshold+0.01),
								 model.types.m.at[0,:].set(cfg.motor_expression_threshold+0.01)])
			return model
		model_factory = _fctry

	# ---

	elif cfg.mdl=="rnd":
		class RandomAgent(eqx.Module):
			logits: jax.Array
			def __init__(self, key):
				self.logits = jr.normal(key, (5,))
			def __call__(self, obs, state, key):
				actions = jnp.array([[0,0],[0,1],[0,-1],[1,0], [-1,0]], dtype=jnp.int16)
				action_id = jr.categorical(key, self.logits)
				action = actions[action_id]
				return action, state
			def initialize(self, key):
				return 0

		model_factory = lambda key: RandomAgent(key)

	# ---

	else:
		raise NameError(f"model {cfg.mdl} is not valid")

	_dummy_model = model_factory(jr.key(0))
	_agent_apply, _agent_init = make_apply_init(_dummy_model)

	if cfg.cast_to_f16:
		def _cast_tree(tree: PyTree):
			return jax.tree.map(
				lambda x: x.astype(jnp.float16) if jnp.issubdtype(x.dtype, jnp.floating) else x,
				tree
			)
		def wrapped_agent_init(prms: jax.Array, key: jax.Array)->PyTree:
			state = _agent_init(prms, key)
			state = _cast_tree(state)
			return state

		def wrapped_agent_apply(prms, obs, state, key):
			obs = _cast_tree(obs)
			action, state = _agent_apply(prms, obs, state, key)
			return action,state

		agent_init = wrapped_agent_init
		agent_apply = wrapped_agent_apply
	else:
		agent_init = _agent_init
		agent_apply = _agent_apply

	return _dummy_model, agent_init, agent_apply

#-------------------------------------------------------------------


def simulate(cfg: Config):

	#-------------------------------------------------------------------

	assert cfg.birth_pool_size<=cfg.max_agents

	#-------------------------------------------------------------------

	key_wrld, key_sim, key_init, key_aux = jr.split(jr.key(cfg.seed), 4)

	#-------------------------------------------------------------------

	dummy_model, agent_init, agent_apply = make_agents_model(cfg)
	dummy_prms = eqx.filter(dummy_model, eqx.is_array)
	prms_shaper = ex.ParameterReshaper(dummy_prms)
	
	#-------------------------------------------------------------------

	if cfg.mdl=="e":
		mutation_fn = partial(mutate, 
							  p_duplicate_split=cfg.p_duplicate_split, 
							  p_duplicate_no_split=cfg.p_duplicate_no_split,
							  p_add=cfg.p_add,
							  p_rm=cfg.p_rm,
							  p_mut=cfg.p_mut,
							  sigma_mut=cfg.sigma_mut,
							  shaper=prms_shaper)

	elif cfg.mdl=="rnd":
		mutation_fn = lambda prms, key: prms + jr.normal(key, prms.shape)*cfg.sigma_mut

	else:
		raise NameError


	_ravel_pytree = lambda x: ravel_pytree(x)[0]

	agent_prms_fctry = lambda key: mutation_fn(_ravel_pytree(
		eqx.filter(dummy_model, eqx.is_array)
	), key)

	#-------------------------------------------------------------------

	if cfg.mdl=="e":

		def _state_energy_cost_fn(state: Agent):
			"""computes state energy cost"""
			net: CTRNN = state.policy_state
			# ---
			assert net.mask is not None
			# ---
			nb_neurons = net.mask.sum()
			s_expressed = jnp.abs(sensory_expression(net, expression_threshold=cfg.sensor_expression_threshold))
			m_expressed = motor_expression(net, 
										   border_threshold=cfg.border_threshold, 
										   expression_threshold=cfg.motor_expression_threshold)
			D = jnp.linalg.norm(net.x[:,None]-net.x[None], axis=-1)
			W = jnp.where(net.mask[None]&net.mask[:,None], net.W, 0.0)
			connection_materials = jnp.abs(W) * D
			total_applied_force = jnp.sum(jnp.clip(net.v*m_expressed, 0.0, cfg.neurons_max_motor_force))

			total_cost = (nb_neurons 			 		  * cfg.neurons_energy_cost
						  + jnp.sum(s_expressed) 		  * cfg.s_expression_energy_cost
						  + jnp.sum(m_expressed) 		  * cfg.m_expression_energy_cost
						  + total_applied_force  		  * cfg.applied_force_energy_cost
						  + jnp.sum(connection_materials) * cfg.connection_cost)

			return (
				total_cost.astype(jnp.float16), 
				{"energy_loss: neurons": nb_neurons, "energy_loss: s_expressed": jnp.sum(s_expressed), 
				 "energy_loss: m_expressed": jnp.sum(m_expressed), "energy_loss: applied_force": total_applied_force,
				 "energy_loss: connection_materials": jnp.sum(connection_materials)}
			)

		state_energy_cost_fn = _state_energy_cost_fn

	else:

		state_energy_cost_fn = lambda state: (jnp.zeros((), jnp.float16), {})

	#-------------------------------------------------------------------

	n = cfg.n_food_types
	
	food_types = FoodType(
		chemical_signature=jnp.identity(n),
		growth_rate=jnp.full((n,), cfg.growth_rate),
		dmin=jnp.full((n,), cfg.min_growth_dist),
		dmax=jnp.full((n,), cfg.max_growth_dist),
		energy_concentration=jnp.full((n,), cfg.energy_concentration),
		spontaneous_grow_prob=jnp.full(n, cfg.spontaneous_grow_prob),
		initial_density=jnp.full(n, cfg.initial_food_density)
	)

	chemical_types = ChemicalType(jnp.full((n,), cfg.diffusion_rate))
	
	world = GridWorld(
		size=cfg.size,
		# ---
		max_agents=cfg.max_agents,
		agent_fctry=agent_prms_fctry,
		agent_apply=agent_apply,
		agent_init=agent_init,
		init_agents=cfg.initial_agents,
		# ---
		reproduction_energy_cost=cfg.reproduction_cost,
		state_energy_cost_fn=state_energy_cost_fn,
		base_energy_loss=cfg.base_energy_loss,
		time_below_threshold_to_die=cfg.time_below_threshold_to_die,
		time_above_threshold_to_reproduce=cfg.time_above_threshold_to_reproduce,
		predation=False,
		passive_eating=cfg.passive_eating,
		passive_reproduction=cfg.passive_reproduction,
		max_energy=cfg.max_energy,
		max_age=cfg.max_age,
		# ---
		mutation_fn=mutation_fn,
		birth_pool_size=cfg.birth_pool_size,
		# ---
		chemical_types=chemical_types,
		food_types=food_types,
		# ---
		walls_density=cfg.wall_density,
		deadly_walls=cfg.deadly_walls,
		# ---
		key=key_wrld
	)

	#-------------------------------------------------------------------

	def count_implicit_types(ctrnn):
		msk = ctrnn.mask > 0.0
		is_sensor = sensory_expression(ctrnn, 
									   expression_threshold=cfg.sensor_expression_threshold).astype(bool)
		is_sensor = jnp.any(is_sensor, -1)
		is_motor = motor_expression(ctrnn,
									border_threshold=cfg.border_threshold,
									expression_threshold=cfg.motor_expression_threshold).astype(bool)
		is_motor = jnp.any(is_motor, -1)
		is_sensorimotor = is_sensor & is_motor
		is_sensor_only = (~is_motor) & is_sensor
		is_motor_only = (~is_sensor) & is_motor
		is_inter = (~(is_sensor|is_motor)) & msk
		return (is_sensor_only.sum(), is_motor_only.sum(), is_sensorimotor.sum(), is_inter.sum())

	def metrics_fn(state: EnvState, step_data: PyTree):
		# ---
		masked_sum = lambda x, mask: jnp.sum(jnp.where(mask, x, 0.0)) # type:ignore
		masked_mean = lambda x, mask: masked_sum(x,mask) / (jnp.sum(mask)+1e-8) #type:ignore
		# ---
		actions = step_data["actions"]
		alive = state.agents.alive
		have_moved = ~jnp.all(actions == jnp.zeros(2, dtype=actions.dtype)[None], axis=-1)
		reproduction_rates = jnp.where(
			alive,
			state.agents.n_offsprings / state.agents.age,
			0.0
		)
		# ---
		observations = step_data["observations"]
		# ---

		if cfg.mdl=="e":
			networks = state.agents.policy_state
			nb_sensors, nb_motors, nb_sensorimotors, nb_inters = jax.vmap(count_implicit_types)(networks)
			prms: PyTree = prms_shaper.reshape(state.agents.prms)
			types_vector = jax.vmap(lambda tree: ravel_pytree(tree)[0])(prms.types)
			active_types = prms.types.active.sum(-1)
			expressed_types = jnp.sum(jnp.round(prms.types.pi * prms.types.active * cfg.N_gain) > 0.0, axis=-1)
			model_metrics = {
				"network_sizes": jnp.where(alive, networks.mask.sum(-1), 0),
				"avg_network_size": masked_mean(networks.mask.sum(-1), alive),
				"nb_sensors": nb_sensors,
				"avg_nb_sensors": masked_mean(nb_sensors, alive),
				"nb_motors": nb_motors,
				"avg_nb_motors": masked_mean(nb_motors, alive),
				"nb_inters": nb_inters,
				"avg_nb_inters": masked_mean(nb_inters, alive),
				"nb_sensorimotors": nb_sensorimotors,
				"avg_nb_sensorimotors": masked_mean(nb_sensorimotors, alive),
				"active_types": active_types,
				"avg_active_types": masked_mean(active_types, alive),
				"expressed_types": expressed_types,
				"avg_expressed_types": masked_mean(expressed_types, alive),
				"types_vector": types_vector
			}
		else:
			model_metrics = {}

		log_data = {
			# --- AGENTS
			"alive": alive,
			"population": alive.sum(),
			"nb_dead": jnp.sum(step_data["dying"]),
			"nb_dead_by_wall": jnp.sum(step_data["dead_by_wall"]),
			"avg_dead_age": step_data["avg_dead_age"],
			"energy_levels": state.agents.energy,
			"nb_above_threshold": masked_sum(state.agents.energy>0.0, alive),
			"nb_below_threshold": masked_sum(state.agents.energy<0.0, alive),
			"avg_energy_levels": masked_mean(state.agents.energy, alive),
			"ages": state.agents.age,
			"avg_age": masked_mean(state.agents.age, alive),
			"generations": state.agents.generation,
			"avg_generation": masked_mean(state.agents.generation, alive),
			"genotypes": state.agents.prms.astype(jnp.float16),
			"offsprings": state.agents.n_offsprings,
			"avg_offsprings": masked_mean(state.agents.n_offsprings, alive), 
			"reproduction_rates": reproduction_rates,
			# --- ACTIONS
			"actions": actions,
			"moving": have_moved,
			"nb_moved": masked_sum(have_moved, alive),
			"nb_reproductions": jnp.sum(step_data["reproducing"]),
			"energy_intakes": step_data["energy_intakes"],
			"avg_energy_intake": masked_mean(step_data["energy_intakes"], alive),
			**{f"avg_{key}": masked_mean(step_data[key], alive) 
			   for key in step_data.keys() if key.startswith("energy_loss")},
			**{key: step_data[key] for key in step_data.keys() if key.startswith("energy_loss")},
			# --- OBS
			"obs_C": observations.chemicals,
			"obs_W": observations.walls,
			# --- NETWORKS
			**model_metrics,
			# --- FOOD
			"total_food": jnp.sum(state.food),
			"total_food_coverage": jnp.sum(state.food>0) / (world.size[0]*world.size[1]),
			**{f"total_food_type_{i}": jnp.sum(food) for i, food in enumerate(state.food)},
			# --- render
			"food_map": state.food,
			"agents_pos": state.agents.position
		}

		log_data = jax.tree.map(lambda x: jnp.where(jnp.isnan(x), jnp.zeros_like(x), x), log_data)

		return log_data, {}, 0

	#-------------------------------------------------------------------

	def host_log_transform(data):
		# ---
		assert wandb.run is not None
		# ---
		data = jax.tree.map(np.asarray, data)

		alive = data["alive"]

		table_fields = []
		fields = list(data.keys())
		for field in fields:
			if data[field].shape and data[field].shape[0]==alive.shape[0]:
				arr = data[field][alive]
				arr = jnp.where(jnp.isnan(arr)|jnp.isinf(arr), 0.0, arr)
				data[field] = arr
				table_fields.append(field)

		log_step = wandb.run._step

		if not log_step % cfg.log_table_freq:

			n_rows = data[table_fields[0]].shape[0]
			table_columns = [data[field] for field in table_fields]
			table_data = [
				[col[r] if field not in ("genotypes","types_vector") 
						else list(col[r]) 
				 for col, field in zip(table_columns, table_fields)]+[log_step] 
			for r in range(n_rows)]
			table = wandb.Table(columns=table_fields+["_step"], data=table_data)
			data["population_data"] = table

		del data["alive"]
		del data["genotypes"]
		del data["generations"]
		del data["moving"]
		del data["types_vector"]
		del data["food_map"]
		del data["agents_pos"]
		del data["actions"]
		del data["obs_C"]
		del data["obs_W"]

		return data


	logger = rx.Logger(cfg.wandb_log, metrics_fn, host_log_transform=host_log_transform)

	#-------------------------------------------------------------------

	def simulation_step(state: EnvState, key: jax.Array):

		state, step_data = world.step(state, key)
		
		logger.log(state, step_data)

		return state

	@partial(jax.jit, static_argnames=("steps"))
	def simulate_n_steps(state: EnvState, key: jax.Array, steps: int):

		def _body_fun(i, c):
			state, key = c
			key, key_step = jr.split(key)
			state = simulation_step(state, key_step)
			return [state, key]

		[state, _] = jax.lax.fori_loop(0, steps, _body_fun, [state, key])

		return state


	#-------------------------------------------------------------------

	state = world.reset(key_init)

	if cfg.wandb_log:
		wandb.init(project="eedx_ediacaran" if not cfg.debug else "DEBUG" , config=cfg._asdict())

	total_env_steps = 0

	while True:

		cmd_and_args = input("cmd: ").strip()
		cmd, *args = [s.strip() for s in cmd_and_args.split(" ")]

		if cmd=="s":
			# do n simulation steps
			if not args:
				print("nb of sim steps not entered")
				continue
			steps = int(args[0])
			key_sim, _key = jr.split(key_sim)
			state = jax.block_until_ready(simulate_n_steps(state, _key, steps))
			total_env_steps += steps

		elif cmd=="ss":
			key_sim, _key = jr.split(key_sim)
			state = simulation_step(state, _key)
		elif cmd=="r":
			fig, ax = plt.subplots()
			world.render(state, ax=ax) #type:ignore
			log = args[0] if args else False
			if log and cfg.wandb_log:
				print("...logging figure to wandb")
				wandb.log({f"env_render: step={total_env_steps}": wandb.Image(fig)}, commit=False)
			plt.show()

		elif cmd=="q":
			break
		else:
			print(f"unknown command {cmd}")

	if cfg.wandb_log:
		wandb.finish()


	return state, {"world": world}



if __name__ == '__main__':
	import matplotlib.pyplot as plt
	import warnings
	warnings.filterwarnings('error', category=FutureWarning)

	cfg = Config(size=(64,64), T_dev=1.0, max_agents=32, initial_agents=16, 
		birth_pool_size=16, max_neurons=64, wandb_log=True, energy_concentration=100.,
		initial_food_density=1.0, mdl="e", cast_to_f16=True, debug=True)
	state, tools = simulate(cfg)
	world = tools["world"]
	plt.show()













