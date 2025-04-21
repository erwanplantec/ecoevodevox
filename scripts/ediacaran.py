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

from src.eco.interface import SpatiallyEmbeddedNetworkInterface
from src.devo.policy_network.rnn import RNNPolicy
from src.devo.policy_network.ctrnn import CTRNNPolicy
from src.eco.gridworld import (Agent, GridWorld,
							   EnvState,
							   FoodType,
							   ChemicalType,
							   Observation)
from src.devo.model_e import (Model_E,  
							  make_mutation_fn,
							  TypeBasedSECTRNN)
from src.devo.grn import GRNEncoding
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
	body_scale: 						float = 1.1
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
	sensor_expression_threshold: 	float = 0.03
	border_threshold: 				float = 0.9
	# --- motor interface
	motor_expression_threshold: float = 0.03
	force_threshold_to_move: 	float = 0.1
	neurons_max_motor_force: 	float = 0.1
	max_total_motor_force:		float = 5.0
	passive_eating: 			bool  = True
	passive_reproduction: 		bool  = True
	# --- dev model
	encoding_mdl: 			str   = "e"
	max_neurons: 			int   = 128
	T_dev: 					float = 10.0
	dt_dev: 				float = 0.1
	n_synaptic_markers: 	int   = 4
	temp_decay: 			float = 0.90
	extra_migration_fields: int   = 3
	N0:						int   = 8
	N_gain: 				float = 50.0
	connection_model: 		str   = "xoxt"
	# --- type based
	max_types: 				int   = 8
	# --- grn based
	regulatory_genes: int=8
	# --- policy
	policy_mdl: str="rnn"
	interface: str="se"
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




#-------------------------------------------------------------------

def make_agents_model(cfg: Config):
	"""
	"""

	sensory_dimensions = cfg.n_food_types + 3
	motor_dimensions = 1 + int(not cfg.passive_eating) + int(not cfg.passive_reproduction)

	if cfg.encoding_mdl=="e":
		def _encoding_mdl__fctry(key):
			
			encoding_model = Model_E(n_types=cfg.max_types, 
						    n_synaptic_markers=cfg.n_synaptic_markers,
						    max_nodes=cfg.max_neurons,
						    sensory_dimensions=sensory_dimensions,
						    motor_dimensions=motor_dimensions,
						    dt=cfg.dt_dev,
						    dvpt_time=cfg.T_dev,
						    temperature_decay=cfg.temp_decay,
						    extra_migration_fields=cfg.extra_migration_fields,
						    N_gain=cfg.N_gain,
						    body_shape="square",
						    key=key)
			encoding_model = eqx.tree_at(lambda p: [p.types.pi,p.types.s, p.types.m, p.connection_model], 
								encoding_model, 
								[encoding_model.types.pi.at[0].set(cfg.N0/cfg.N_gain),
								 encoding_model.types.s.at[0,:].set(cfg.sensor_expression_threshold+0.01),
								 encoding_model.types.m.at[0,:].set(cfg.motor_expression_threshold+0.01),
								 jax.tree.map(
								 	lambda x: jnp.zeros_like(x) if eqx.is_array(x) else x,
								 	encoding_model.connection_model)
								])
			
			return encoding_model

		encoding_model_factory = _encoding_mdl__fctry

	elif cfg.encoding_mdl=="grn":

		def _grn_mdl_fctry(key):
			mdl = GRNEncoding(sensory_dimensions, motor_dimensions, cfg.n_synaptic_markers, cfg.regulatory_genes,
				cfg.extra_migration_fields, cfg.dt_dev, cfg.T_dev, cfg.N_gain, cfg.max_neurons, key=key)
			mdl = eqx.tree_at(lambda x: x.population, mdl, jnp.full_like(mdl.population, cfg.N0/cfg.N_gain))
			return mdl
		encoding_model_factory = _grn_mdl_fctry

	# ---

	else:
		raise NameError(f"model {cfg.encoding_mdl} is not valid")

	#-------------------------------------------------------------------

	if cfg.interface=="se":
		interface = SpatiallyEmbeddedNetworkInterface(cfg.sensor_expression_threshold,
													  jnn.tanh,
													  cfg.motor_expression_threshold,
													  border_threshold=cfg.border_threshold,
													  max_neuron_force=cfg.neurons_max_motor_force,
													  force_threshold_to_move=cfg.force_threshold_to_move,
													  max_motor_force=cfg.max_total_motor_force)
	else: 
		raise NameError(f"{cfg.interface} is not a valid interface")


	if cfg.policy_mdl=="rnn":
		def _rnn_mdl_factory(key):
			encoding_mdl = encoding_model_factory(key)
			mdl = RNNPolicy(encoding_mdl, interface)
			return mdl
		model_factory = _rnn_mdl_factory
	elif  cfg.policy_mdl=="ctrnn":
		def _ctrnn_mdl_factory(key):
			encoding_mdl = encoding_model_factory(key)
			mdl = CTRNNPolicy(encoding_mdl, interface, cfg.dt_ctrnn, cfg.T_ctrnn)
			return mdl
		model_factory = _ctrnn_mdl_factory
	else:
		raise NameError(f"no policy model: {cfg.policy_mdl}")

	#-------------------------------------------------------------------

	_dummy_model = model_factory(jr.key(0))
	_agent_apply, _agent_init = make_apply_init(_dummy_model, reshape_prms=False)

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
	interface = dummy_model.interface
	dummy_prms = eqx.filter(dummy_model, eqx.is_array)
	flat_prms_like, prms_shaper = ravel_pytree(dummy_prms)
	
	#-------------------------------------------------------------------

	if cfg.encoding_mdl=="e":
		dummy_encoder_prms = dummy_prms.encoding_model
		encoder_mutation_fn = make_mutation_fn(dummy_encoder_prms, 
											   p_duplicate_split=cfg.p_duplicate_split, 
											   p_duplicate_no_split=cfg.p_duplicate_no_split,
											   p_add=cfg.p_add,
											   p_rm=cfg.p_rm,
											   p_mut=cfg.p_mut,
											   sigma_mut=cfg.sigma_mut)
		
		def _e_mutation_fn(prms: PyTree, key: jax.Array):
			encoder_prms = encoder_mutation_fn(prms.encoding_model, key)
			prms  = eqx.tree_at(lambda tree: tree.encoding_model, prms, encoder_prms)
			return prms
		mutation_fn = _e_mutation_fn

	elif cfg.encoding_mdl in ["grn"]:

		def _grn_mutation_fn(prms: PyTree, key: jax.Array):
			epsilon = prms_shaper(jr.normal(key, flat_prms_like.shape)*cfg.sigma_mut)
			prms = jax.tree.map(lambda x,e:x+e, prms, epsilon)
			return prms
		mutation_fn = _grn_mutation_fn

	elif cfg.encoding_mdl in ["rnd"]:
		mutation_fn = lambda prms, key: prms + jr.normal(key, prms.shape)*cfg.sigma_mut

	else:
		raise NameError


	agent_prms_fctry = lambda key: mutation_fn(
		eqx.filter(dummy_model, eqx.is_array), key
	)

	#-------------------------------------------------------------------

	if cfg.encoding_mdl=="e":

		def _state_energy_cost_fn(state: Agent):
			"""computes state energy cost"""
			net: TypeBasedSECTRNN = state.policy_state
			# ---
			assert net.mask is not None
			assert isinstance(interface, SpatiallyEmbeddedNetworkInterface)
			# ---
			nb_neurons = net.mask.sum()
			s_expressed = jnp.abs(interface.sensory_expression(net))
			m_expressed = jnp.abs(interface.motor_expression(net))
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
		agents_scale=cfg.body_scale,
		reproduction_energy_cost=cfg.reproduction_cost,
		state_energy_cost_fn=state_energy_cost_fn,
		base_energy_loss=cfg.base_energy_loss,
		time_below_threshold_to_die=cfg.time_below_threshold_to_die,
		time_above_threshold_to_reproduce=cfg.time_above_threshold_to_reproduce,
		predation=False,
		passive_eating=cfg.passive_eating,
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
		assert isinstance(interface, SpatiallyEmbeddedNetworkInterface)
		msk = ctrnn.mask > 0.0
		is_sensor = interface.sensory_expression(ctrnn).astype(bool)
		is_sensor = jnp.any(is_sensor, -1)
		is_motor = interface.motor_expression(ctrnn).astype(bool)
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

		if cfg.encoding_mdl in["e","grn"]:
			networks = state.agents.policy_state
			nb_sensors, nb_motors, nb_sensorimotors, nb_inters = jax.vmap(count_implicit_types)(networks)
			prms: PyTree = state.agents.prms
			enc_prms = prms.encoding_model
			network_sizes = jnp.where(alive, networks.mask.sum(-1), 0); assert isinstance(network_sizes, jax.Array)

			right_neural_density  = jnp.where((networks.x[:,:,0]> cfg.border_threshold)&(networks.mask), 1, 0).sum(-1)
			left_neural_density   = jnp.where((networks.x[:,:,0]<-cfg.border_threshold)&(networks.mask), 1, 0).sum(-1)
			top_neural_density    = jnp.where((networks.x[:,:,1]> cfg.border_threshold)&(networks.mask), 1, 0).sum(-1)
			bottom_neural_density = jnp.where((networks.x[:,:,1]<-cfg.border_threshold)&(networks.mask), 1, 0).sum(-1)

			lateralization = jnp.max(
				jnp.stack([right_neural_density, left_neural_density, top_neural_density, bottom_neural_density], axis=-1),
				axis=-1
			)
			lateralization = jnp.where(network_sizes>0, lateralization/network_sizes, 0)

			make_impossible_moves = ((right_neural_density ==0) & (state.agents.move_left_count >0)
								    |(left_neural_density  ==0) & (state.agents.move_right_count>0)
								    |(top_neural_density   ==0) & (state.agents.move_down_count >0)
								    |(bottom_neural_density==0) & (state.agents.move_up_count   >0))

			model_metrics = {
				"network_sizes": network_sizes,
				"right_neural_density": right_neural_density,
				"left_neural_density": left_neural_density,
				"top_neural_density": top_neural_density,
				"bottom_neural_density": bottom_neural_density,
				"nb_sensors": nb_sensors,
				"nb_motors": nb_motors,
				"nb_inters": nb_inters,
				"nb_sensorimotors": nb_sensorimotors,
				"make_impossible_move": make_impossible_moves,
			}
			if cfg.encoding_mdl=="e":
				types_vector = jax.vmap(lambda tree: ravel_pytree(tree)[0])(enc_prms.types)
				active_types = enc_prms.types.active.sum(-1)
				expressed_types = jnp.sum(jnp.round(enc_prms.types.pi * enc_prms.types.active * cfg.N_gain) > 0.0, axis=-1)
				model_metrics["active_types"] = active_types
				model_metrics["expressed_types"] = expressed_types
				model_metrics["types_vector"] = types_vector
		else:
			model_metrics = {}

		log_data = {
			# --- AGENTS
			"alive": alive,
			"population": alive.sum(),
			"nb_dead": jnp.sum(step_data["dying"]),
			"avg_dead_age": step_data["avg_dead_age"],
			"energy_levels": state.agents.energy,
			"ages": state.agents.age,
			"generations": state.agents.generation,
			"genotypes": jax.vmap(lambda tree: ravel_pytree(tree)[0])(state.agents.prms),
			"offsprings": state.agents.n_offsprings,
			"avg_offsprings": masked_mean(state.agents.n_offsprings, alive), 
			"reproduction_rates": reproduction_rates,
			# --- ACTIONS
			"actions": actions,
			"actions_norm": jnp.linalg.norm(actions, axis=-1),
			"moving": have_moved,
			"move_up_count": state.agents.move_up_count,
			"move_down_count": state.agents.move_down_count,
			"move_right_count": state.agents.move_right_count,
			"move_left_count": state.agents.move_left_count,
			"nb_reproductions": jnp.sum(step_data["reproducing"]),
			"energy_intakes": step_data["energy_intakes"],
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

		if alive.sum()==0:
			return {"population":0}

		table_fields = []
		fields = list(data.keys())
		for field in fields:
			if data[field].shape and data[field].shape[0]==alive.shape[0]:
				arr = data[field][alive]
				#arr = np.where(np.isnan(arr)|np.isinf(arr), 0.0, arr)
				data[field] = arr
				table_fields.append(field)
				
				if arr.ndim==1:
					data[f"{field} (avg)"] = np.mean(arr)
					data[f"{field} (max)"] = np.max(arr)
					data[f"{field} (min)"] = np.min(arr)

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

		for field in table_fields:
			del data[field]

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
		wandb.init(project=cfg.wandb_project if not cfg.debug else "DEBUG" , config=cfg._asdict())

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
		initial_food_density=1.0, encoding_mdl="grn", cast_to_f16=True, debug=True, body_scale=10,
		policy_mdl="rnn")
	state, tools = simulate(cfg)
	world = tools["world"]
	plt.show()





