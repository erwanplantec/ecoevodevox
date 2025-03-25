from functools import partial
from jax.flatten_util import ravel_pytree
from jaxtyping import PyTree
import realax as rx
import evosax as ex
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import numpy as np
import equinox as eqx
import matplotlib
import matplotlib.pyplot as plt
from celluloid import Camera
import wandb

from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.utils.plotting import plot_2d_map_elites_repertoire

from src.devo.ctrnn import CTRNN, CTRNNPolicyConfig
from src.devo.model_e import Model_E, make_single_type, mutate, mask_prms, min_prms, max_prms

from typing import NamedTuple

class Config(NamedTuple):
	# --- training ---
	seed: int=0
	log: bool=True
	batch_size: int=256
	grid_shape: tuple=(32,32)
	plot_freq: int=100
	make_animation: bool|str="ask"
	# --- env ---
	has_target: bool=False
	maze: str="standard"
	connection_cost: float=0.0
	neuron_cost: float=0.0
	sensor_cost: float=0.0
	motor_cost: float=0.0
	# --- robot ---
	lasers: int=32
	laser_ranges: float=0.2
	sensor_neurons_min_norm: float=0.8
	motor_neurons_min_norm: float=0.8
	motor_neurons_force: float=0.1
	# --- model ---
	max_types: int=8
	max_nodes: int=128
	synaptic_markers: int=8
	N_gain: float=100.0
	conn_model="mlp"
	# --- mutations ---
	sigma_mut: float=0.01
	p_duplicate: float=0.005
	p_mut: float=1.0
	p_rm: float=0.005
	p_add: float=0.005
	# --- variation ---
	variation_mode: str="isoline"
	iso_sigma: float=0.01
	line_sigma: float=0.01
	variation_percentage: float=0.5

def train(cfg: Config):

	key = jr.key(cfg.seed)
	key_train, key_init, key_mdl = jr.split(key, 3)

	angle_step = 2*jnp.pi / cfg.lasers
	laser_angles = jnp.linspace(-jnp.pi, jnp.pi - angle_step, cfg.lasers)
	laser_positions = jnp.concatenate(
		[jnp.sin(laser_angles)[:,None],
		 jnp.cos(laser_angles)[:,None]], axis=-1)

	def encode_fn(ctrnn: CTRNN, obs: jax.Array):
		# ---
		assert ctrnn.s is not None
		# ---
		laser_values = obs[:cfg.lasers]
		laser_values = 1.0 - (laser_values/cfg.laser_ranges)
		xs = ctrnn.x
		xs_norm = jnp.linalg.norm(xs, axis=-1)
		is_on_border = xs_norm>cfg.sensor_neurons_min_norm
		dists = jnp.linalg.norm(xs[:,None] - laser_positions, axis=-1)
		closest = jnp.argmin(dists, axis=-1)
		I = jnp.where(
			is_on_border,
			laser_values[closest] * jnn.sigmoid(ctrnn.s[:,0]*2.0),
			jnp.zeros_like(ctrnn.v)
		)
		return I

	def decode_fn(ctrnn: CTRNN):
		# ---
		assert ctrnn.m is not None
		# ---
		xs_x = ctrnn.x[:,0]

		on_left_motor = jnp.where(xs_x < -cfg.motor_neurons_min_norm, 
							 	  jnn.sigmoid(ctrnn.m[:,0]*2.0)*ctrnn.v*cfg.motor_neurons_force, 
								  0.0)

		on_right_motor = jnp.where(xs_x > cfg.motor_neurons_min_norm, 
							 	   jnn.sigmoid(ctrnn.m[:,0]*2.0)*ctrnn.v*cfg.motor_neurons_force, 
								   0.0)

		action = jnp.array([on_left_motor.sum(), on_right_motor.sum()])

		return action


	policy_cfg = CTRNNPolicyConfig(encode_fn, decode_fn)
	model = Model_E(cfg.max_types, cfg.synaptic_markers, cfg.max_nodes, 
		sensory_dimensions=2, motor_dimensions=1, temperature_decay=0.98, 
		extra_migration_fields=3, N_gain=cfg.N_gain, body_shape="square", policy_cfg=policy_cfg,
		connection_model=cfg.conn_model, key=key_mdl)
	model = make_single_type(model, 8)
	
	prms, sttcs = model.partition()
	prms_shaper = ex.ParameterReshaper(prms)
	mdl_fctry = lambda prms: eqx.combine(prms, sttcs) 
	
	prms_min, _ = ravel_pytree(min_prms(prms)) 
	prms_max, _ = ravel_pytree(max_prms(prms))
	prms_mask, _ = ravel_pytree(mask_prms(prms))

	_mutation_fn = partial(
		mutate, 
		p_duplicate=cfg.p_duplicate, 
		p_mut=cfg.p_mut, 
		p_rm=cfg.p_rm, 
		p_add=cfg.p_add, 
		sigma_mut=cfg.sigma_mut,
		shaper=prms_shaper
	)
	def mutation_fn(x_batch, key):
		new_key, key = jr.split(key)
		keys = jr.split(key, x_batch.shape[0])
		return jax.vmap(_mutation_fn)(x_batch, keys), new_key

	_isoline_variation = partial(
		isoline_variation,
		iso_sigma=cfg.iso_sigma,
		line_sigma=cfg.line_sigma,
		minval=prms_min, #type:ignore
		maxval=prms_max #type:ignore
	)
	def _crossover(x1, x2, key):
		prms1 = prms_shaper.reshape_single(x1)
		prms2 = prms_shaper.reshape_single(x2)

		type_selec = jr.choice(key, 2, shape=(cfg.max_types,))
		types = jax.tree.map(
			lambda xs: jnp.stack([x[...,i] for x, i in zip(xs, type_selec)]), 
			jax.tree.map(lambda a,b: jnp.stack([a,b], axis=-1), prms1.types, prms2.types)
		)
		order = jnp.argsort(types.active, descending=True)
		types = jax.tree.map(lambda x: x[order], types)
		prms = eqx.tree_at(lambda p: p.types, prms1, types)
		x = prms_shaper.flatten_single(prms)
		return x, key

	def variation_fn(x1, x2, key):
		key, new_key = jr.split(key)
		if cfg.variation_mode=="cross":
			x_varied, _ = jax.vmap(_crossover)(x1, x2, jr.split(key, x1.shape[0]))
		elif cfg.variation_mode=="isoline":
			x_varied, _ = _isoline_variation(x1, x2, key)
		x_varied = jnp.where(prms_mask.astype(bool), x_varied, x1) #type:ignore
		return x_varied, new_key

	emitter = MixingEmitter(mutation_fn, variation_fn, cfg.variation_percentage, cfg.batch_size) #type:ignore

	robot_kwargs = dict(laser_angles=laser_angles, laser_ranges=cfg.laser_ranges)

	_task = rx.KheperaxTask(maze=cfg.maze, model_factory=mdl_fctry, 
		robot_kwargs=robot_kwargs, has_target=cfg.has_target)
	
	def task(prms, key, _=None):
		fitness, data = _task(prms, key)
		final_state = data["final_state"]
		final_pos = final_state.env_state.robot.posture
		bd = jnp.array([final_pos.x, final_pos.y])
		
		policy_state = final_state.policy_state

		xs = policy_state.x
		D = jnp.linalg.norm(xs[None]-xs[:,None], axis=-1)
		connections = (jnp.abs(policy_state.W) * D).sum()
		nb_neurons = policy_state.mask.sum()
		sensors = (jnn.sigmoid(policy_state.s*2.0) * policy_state.mask[:,None]).sum()
		motors = (jnn.sigmoid(policy_state.m*2.0) * policy_state.mask[:,None]).sum()

		connections_penalty = connections * cfg.connection_cost
		neurons_penalty = nb_neurons * cfg.neuron_cost
		sensors_penalty = sensors * cfg.sensor_cost
		motors_penalty = motors * cfg.motor_cost

		fitness = fitness - connections_penalty - neurons_penalty - sensors_penalty - motors_penalty

		return fitness, bd, data

	#-------------------------------------------------------------------
	#-------------------------------------------------------------------
	#-------------------------------------------------------------------

	def metrics_fn(state, data):

		repertoire = state.repertoire
		genotypes = repertoire.genotypes
		mask = ~jnp.isinf(repertoire.fitnesses)
		
		prms = prms_shaper.reshape(genotypes)
		active_types = prms.types.active.sum(-1).astype(int) #type:ignore

		nb_types_coverage = {
			f"{k}-coverage": jnp.where(mask&(active_types==k), 1.0, 0.0).mean()
		for k in range(1, cfg.max_types+1)}

		log_data = dict(
			repertoire=repertoire,
			coverage = jnp.where(mask, 1.0, 0.0).mean(),
			max_fitness = jnp.max(repertoire.fitnesses),
			fitnesses = repertoire.fitnesses,
			qd = jnp.sum(jnp.where(mask, repertoire.fitnesses, 0.0)), #type:ignore
			avg_active_types=jnp.sum(jnp.where(mask, active_types, 0.0)) / mask.sum(), #type:ignore
			active_types=jnp.where(mask, active_types, 0.0), #type:ignore
			max_active_types = active_types.max(), #type:ignore
			network_size = jnp.where(mask, jnp.sum(prms.types.active*prms.types.pi*cfg.N_gain, axis=-1), 0.0), #type:ignore
			**nb_types_coverage
		)

		return log_data, None, 0

	repertoires = []
	generations = []
	counter = [0]

	def host_transform(data):
		#data = jax.tree.map(np.asarray, data)
		counter[0] += 1
		if not counter[0]%cfg.plot_freq:
			generations.append(counter[0])
			repertoires.append(jax.tree.map(np.asarray, data["repertoire"]))
		del data["repertoire"]
		mask = ~np.isinf(data["fitnesses"])
		data["active_types"] = data["active_types"][mask]
		data["network_size"] = data["network_size"][mask]
		data["fitnesses"] = data["fitnesses"][mask]
		return data
	logger = rx.Logger(cfg.log, metrics_fn=metrics_fn, host_log_transform=host_transform)

	trainer = rx.QDTrainer(emitter, task, 1, params_like=prms, bd_minval=0.0, bd_maxval=1.0, grid_shape=cfg.grid_shape, logger=logger) 
	
	#-------------------------------------------------------------------
	#-------------------------------------------------------------------
	#-------------------------------------------------------------------

	k_eval, k_init, k_emit = jr.split(key_init, 3)
	x_init = prms_shaper.flatten_single(prms)
	init_genotypes = jax.vmap(_mutation_fn, in_axes=(None,0))(x_init,jr.split(k_init, cfg.batch_size)) #type:ignore
	init_fitnesses, init_bds, _ = trainer.eval(init_genotypes, k_eval, None)
	repertoire = MapElitesRepertoire.init(init_genotypes, init_fitnesses, init_bds, trainer.centroids)
	emitter_state, _ = trainer.emitter.init(k_emit, repertoire, init_genotypes, init_fitnesses, init_bds, None) #type:ignore
	init_state = rx.training.qd.QDState(repertoire=repertoire, emitter_state=emitter_state)
	
	if cfg.log: wandb.init(project="eedx_qd", config=cfg._asdict())
	state = None
	while True:
		gens = input("Training generations:")
		if not gens: 
			break
		try: 
			gens = int(gens)
		except: 
			continue
		trainer.train_steps = int(gens)
		state = jax.block_until_ready(trainer.train_(init_state, key_train))

		fig, ax = plt.subplots(1, 3, figsize=(18,6), sharey=True)
		repertoire = state.repertoire
		fitnesses = repertoire.fitnesses
		genotypes = repertoire.genotypes
		mask = ~np.isinf(fitnesses)
		prms: PyTree = prms_shaper.reshape(genotypes)
		active_types = prms.types.active.sum(-1)
		network_size = np.sum(prms.types.active * prms.types.pi * cfg.N_gain, axis=-1)

		plot_2d_map_elites_repertoire(trainer.centroids, fitnesses, minval=0.0, maxval=1.0, ax=ax[0]) #type:ignore
		ax[0].set_title("fitness")
		plot_2d_map_elites_repertoire(trainer.centroids, np.where(mask, active_types, -jnp.inf), minval=0.0, maxval=1.0, ax=ax[1])
		ax[1].set_title("#types")
		ax[1].set_ylabel("")
		plot_2d_map_elites_repertoire(trainer.centroids, np.where(mask, network_size, -jnp.inf), minval=0.0, maxval=1.0, ax=ax[2])
		ax[2].set_title("N")
		ax[2].set_ylabel("")
		fig.tight_layout()
		if cfg.log: wandb.log(dict(final_result=wandb.Image(fig)))
		plt.show()

	#-------------------------------------------------------------------

	if cfg.make_animation=="ask":
		make_animation=input("make animation ?")
		make_animation = make_animation in ["y", "Y"]
	else:
		make_animation = cfg.make_animation

	if make_animation:
		fig, ax = plt.subplots(1, 3, figsize=(18,6), sharey=True)
		cam = Camera(fig)
		max_fitness = max([r.fitnesses.max() for r in repertoires])
		min_fitness = min([np.where(np.isinf(r.fitnesses), np.inf, r.fitnesses).min() for r in repertoires])
		max_types = max([np.max(prms_shaper.reshape(r.genotypes).types.active.sum(-1)) for r in repertoires])
		min_types = 0
		for repertoire in repertoires:
			fitnesses = repertoire.fitnesses
			genotypes = repertoire.genotypes
			mask = ~np.isinf(fitnesses)
			prms: PyTree = prms_shaper.reshape(genotypes)
			active_types = prms.types.active.sum(-1)
			network_size = np.sum(prms.types.active * prms.types.pi * cfg.N_gain, axis=-1)

			plot_2d_map_elites_repertoire(trainer.centroids, fitnesses, minval=0.0, maxval=1.0, ax=ax[0], vmin=min_fitness, vmax=max_fitness) #type:ignore
			ax[0].set_title("fitness")
			plot_2d_map_elites_repertoire(trainer.centroids, np.where(mask, active_types, -jnp.inf), minval=0.0, maxval=1.0, ax=ax[1], vmin=min_types, vmax=max_types)
			ax[1].set_title("#types")
			ax[1].set_ylabel("")
			plot_2d_map_elites_repertoire(trainer.centroids, np.where(mask, network_size, -jnp.inf), minval=0.0, maxval=1.0, ax=ax[2])
			ax[2].set_title("N")
			ax[2].set_ylabel("")

			cam.snap()
		ani = cam.animate()
		if cfg.log: wandb.log({"result": wandb.Html(ani.to_html5_video())}, commit=False)

	if cfg.log: wandb.finish()

	return state


if __name__ == '__main__':
	cfg = Config(batch_size=16, N_gain=100, p_duplicate=0.01, variation_percentage=0.3, plot_freq=5, variation_mode="cross", log=True)
	train(cfg)







