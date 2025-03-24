from functools import partial
from jax.flatten_util import ravel_pytree
import realax as rx
import evosax as ex
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import equinox as eqx
import matplotlib.pyplot as plt

from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.utils.plotting import plot_2d_map_elites_repertoire

from src.devo.ctrnn import CTRNN, CTRNNPolicyConfig
from src.devo.model_e import Model_E, make_single_type, mutate, mask_prms, min_prms, max_prms

from typing import NamedTuple

class Config(NamedTuple):
	# --- training ---
	gens: int=256
	batch_size: int=256
	# --- robot ---
	lasers: int=32
	sensor_neurons_min_norm: float=0.8
	motor_neurons_min_norm: float=0.8
	# --- model ---
	max_types: int=8
	max_nodes: int=128
	synaptic_markers: int=8
	N_gain: float=100.0
	# --- mutations ---
	sigma_mut: float=0.01
	p_duplicate: float=0.005
	p_mut: float=1.0
	p_rm: float=0.005
	p_add: float=0.005
	# --- variation ---
	iso_sigma: float=0.01
	line_sigma: float=0.01
	variation_percentage: float=0.5

def train(cfg: Config, key: jax.Array):

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
		xs = ctrnn.x
		xs_norm = jnp.linalg.norm(xs, axis=-1)
		is_on_border = xs_norm>cfg.sensor_neurons_min_norm
		dists = jnp.linalg.norm(xs[:,None] - laser_positions, axis=-1)
		closest = jnp.argmin(dists, axis=-1)
		I = jnp.where(
			is_on_border,
			laser_values[closest] * ctrnn.s[:,0],
			jnp.zeros_like(ctrnn.v)
		)
		return I

	def decode_fn(ctrnn: CTRNN):
		# ---
		assert ctrnn.m is not None
		# ---
		xs_x = ctrnn.x[:,0]

		on_left_motor = jnp.where(xs_x < -cfg.motor_neurons_min_norm, 
							 	  ctrnn.m[:,0]*ctrnn.v, 
								  0.0)

		on_right_motor = jnp.where(xs_x > cfg.motor_neurons_min_norm, 
							 	   ctrnn.m[:,0]*ctrnn.v, 
								   0.0)

		action = jnp.array([on_left_motor.sum(), on_right_motor.sum()])

		return action



	policy_cfg = CTRNNPolicyConfig(encode_fn, decode_fn)
	model = Model_E(cfg.max_types, cfg.synaptic_markers, cfg.max_nodes, 
		sensory_dimensions=2, motor_dimensions=1, temperature_decay=0.98, 
		extra_migration_fields=3, N_gain=cfg.N_gain, body_shape="circle", policy_cfg=policy_cfg,
		key=key_mdl)
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

	_variation_fn = partial(
		isoline_variation,
		iso_sigma=cfg.iso_sigma,
		line_sigma=cfg.line_sigma,
		minval=prms_min, #type:ignore
		maxval=prms_max #type:ignore
	)
	def variation_fn(x1, x2, key):
		x_varied, key = _variation_fn(x1, x2, key)
		x_varied = jnp.where(prms_mask.astype(bool), x_varied, x1) #type:ignore
		return x_varied, key

	emitter = MixingEmitter(mutation_fn, variation_fn, cfg.variation_percentage, cfg.batch_size) #type:ignore

	robot_kwargs = dict(laser_angles=laser_angles)

	_task = rx.KheperaxTask("standard", model_factory=mdl_fctry, robot_kwargs=robot_kwargs)
	
	def task(prms, key, _=None):
		fitness, data = _task(prms, key)
		final_state = data["final_state"]
		final_pos = final_state.env_state.robot.posture
		bd = jnp.array([final_pos.x, final_pos.y])
		return fitness, bd, data

	trainer = rx.QDTrainer(emitter, task, cfg.gens, params_like=prms, bd_minval=0.0, bd_maxval=1.0, grid_shape=(64,64)) 
	
	#-------------------------------------------------------------------
	#-------------------------------------------------------------------
	#-------------------------------------------------------------------

	k_eval, k_init, k_emit = jr.split(key_init, 3)
	x_init = prms_shaper.flatten_single(prms)
	init_genotypes = jax.vmap(_mutation_fn, in_axes=(None,0))(x_init,jr.split(k_init, cfg.batch_size)) #type:ignore
	init_fitnesses, init_bds, _ = trainer.eval(init_genotypes, k_eval, None)
	repertoire = MapElitesRepertoire(init_genotypes, init_fitnesses, init_bds, trainer.centroids)
	emitter_state, _ = trainer.emitter.init(k_emit, repertoire, init_genotypes, init_fitnesses, init_bds, None)
	init_state = rx.training.qd.QDState(repertoire=repertoire, emitter_state=emitter_state)
	
	state = trainer.train_(init_state, key_train)

	plot_2d_map_elites_repertoire(trainer.centroids, state.repertoire.fitnesses, minval=0.0, maxval=1.0, repertoire_descriptors=state.repertoire.descriptors)

	plt.show()


if __name__ == '__main__':
	cfg = Config(batch_size=256, gens=128, N_gain=100, p_duplicate=0.01, variation_percentage=0.1)
	train(cfg, jr.key(1))







