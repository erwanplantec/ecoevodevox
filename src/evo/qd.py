import jax
import jax.numpy as jnp
import jax.random as jr
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire, get_cells_indices, Centroid
from qdax.core.containers.mels_repertoire import MELSRepertoire, Genotype, Descriptor, Fitness, ExtraScores, _dispersion

class GreedyMELSRepertoire(MELSRepertoire):
	@jax.jit
	def add(
		self,
		batch_of_genotypes: Genotype,
		batch_of_descriptors: Descriptor,
		batch_of_fitnesses: Fitness,
		batch_of_extra_scores: ExtraScores|None = None,
	):	
		batch_size, num_samples = batch_of_fitnesses.shape

		# Compute indices/cells of all descriptors.
		batch_of_indices = get_cells_indices(
			batch_of_descriptors.reshape(batch_size * num_samples, -1), self.centroids
		) #(P*S,)


		# Compute dispersion / spread. The dispersion is set to zero if
		# num_samples is 1.
		batch_of_spreads = jax.lax.cond(
			num_samples == 1,
			lambda desc: jnp.zeros(batch_size),
			lambda desc: jax.vmap(_dispersion)(
				desc.reshape((batch_size, num_samples, -1))
			),
			batch_of_descriptors,
		)
		batch_of_spreads = jnp.expand_dims(batch_of_spreads, axis=-1).repeat(num_samples, -1).ravel()
		batch_of_descriptors = batch_of_descriptors.reshape((batch_size*num_samples, -1))

		# Compute canonical descriptors as the descriptor of the centroid. Note that this line redefines the earlier batch_of_descriptors.
		# batch_of_descriptors = jnp.take_along_axis(
		# 	self.centroids, batch_of_indices, axis=0
		# )

		batch_of_fitnesses = batch_of_fitnesses.ravel()

		num_centroids = self.centroids.shape[0]

		# get current repertoire fitnesses and spreads
		current_fitnesses = jnp.take_along_axis(
			self.fitnesses, batch_of_indices, 0
		)


		current_spreads = jnp.take_along_axis(self.spreads, batch_of_indices, 0)

		# get addition condition
		addition_condition_fitness = batch_of_fitnesses > current_fitnesses
		addition_condition_spread = batch_of_spreads <= current_spreads
		addition_condition = jnp.logical_and(
			addition_condition_fitness, addition_condition_spread
		)

		# assign fake position when relevant : num_centroids is out of bound
		batch_of_indices = jnp.where(
			addition_condition, batch_of_indices, num_centroids
		)


		# create new repertoire
		batch_of_genotypes = jax.tree.map(
			lambda x: x[:,None].repeat(num_samples,1).reshape(-1,*x.shape[1:]),
			batch_of_genotypes
		)
		new_repertoire_genotypes = jax.tree_util.tree_map(
			lambda repertoire_genotypes, new_genotypes: repertoire_genotypes.at[
				batch_of_indices
			].set(new_genotypes),
			self.genotypes,
			batch_of_genotypes,
		)

		# compute new fitness and descriptors
		new_fitnesses = self.fitnesses.at[batch_of_indices].set(
			batch_of_fitnesses
		)
		new_descriptors = self.descriptors.at[batch_of_indices].set(
			batch_of_descriptors
		)
		new_spreads = self.spreads.at[batch_of_indices].set(
			batch_of_spreads
		)

		return MELSRepertoire(
			genotypes=new_repertoire_genotypes,
			fitnesses=new_fitnesses,
			descriptors=new_descriptors,
			centroids=self.centroids,
			spreads=new_spreads,
		)

def _compute_scores(bds: jax.Array, s: float):
	dists = jnp.linalg.norm(jnp.square(bds[:,None] - bds[None]), axis=-1)
	contribs = jnp.exp(-dists/s)
	return contribs.sum(0)

class IlluminatePotentialRepertoire(MapElitesRepertoire):
	
	scores: jax.Array
	sigma: float
	
	@jax.jit
	def add(
		self,
		batch_of_genotypes: Genotype,
		batch_of_descriptors: Descriptor,
		batch_of_fitnesses: Fitness,
		batch_of_extra_scores: ExtraScores|None = None,
	):
		batch_size, num_samples = batch_of_fitnesses.shape

		# Compute indices/cells of all descriptors.
		batch_of_indices = get_cells_indices(
			batch_of_descriptors.reshape(batch_size * num_samples, -1), self.centroids
		) #(P*S,)


		# Compute dispersion / spread. The dispersion is set to zero if
		# num_samples is 1.
		batch_of_scores = jax.lax.cond(
			num_samples == 1,
			lambda desc: jnp.zeros((batch_size, num_samples)),
			lambda desc: jax.vmap(_compute_scores, in_axes=(0,None))(
				desc.reshape((batch_size, num_samples, -1)), self.sigma
			),
			batch_of_descriptors,
		)
		batch_of_scores = batch_of_scores.ravel()
		batch_of_descriptors = batch_of_descriptors.reshape((batch_size*num_samples, -1))

		# Compute canonical descriptors as the descriptor of the centroid. Note that this line redefines the earlier batch_of_descriptors.
		# batch_of_descriptors = jnp.take_along_axis(
		# 	self.centroids, batch_of_indices, axis=0
		# )

		batch_of_fitnesses = batch_of_fitnesses.ravel()

		num_centroids = self.centroids.shape[0]

		# get current repertoire fitnesses and spreads
		current_fitnesses = jnp.take_along_axis(
			self.fitnesses, batch_of_indices, 0
		)


		current_scores = jnp.take_along_axis(self.scores, batch_of_indices, 0)

		# get addition condition
		addition_condition_fitness = batch_of_fitnesses > current_fitnesses
		addition_condition_spread = batch_of_scores >= current_scores
		addition_condition = jnp.logical_and(
			addition_condition_fitness, addition_condition_spread
		)

		# assign fake position when relevant : num_centroids is out of bound
		batch_of_indices = jnp.where(
			addition_condition, batch_of_indices, num_centroids
		)

		# create new repertoire
		batch_of_genotypes = jax.tree.map(
			lambda x: x[:,None].repeat(num_samples,1).reshape(-1,*x.shape[1:]),
			batch_of_genotypes
		)
		new_repertoire_genotypes = jax.tree_util.tree_map(
			lambda repertoire_genotypes, new_genotypes: repertoire_genotypes.at[
				batch_of_indices
			].set(new_genotypes),
			self.genotypes,
			batch_of_genotypes,
		)

		# compute new fitness and descriptors
		new_fitnesses = self.fitnesses.at[batch_of_indices].set(
			batch_of_fitnesses
		)
		new_descriptors = self.descriptors.at[batch_of_indices].set(
			batch_of_descriptors
		)
		new_scores = self.scores.at[batch_of_indices].set(
			batch_of_scores
		)

		return IlluminatePotentialRepertoire(
			genotypes=new_repertoire_genotypes,
			fitnesses=new_fitnesses,
			descriptors=new_descriptors,
			centroids=self.centroids,
			scores=new_scores,
			sigma=self.sigma
		)

	@classmethod
	def init(cls, 
		genotypes: Genotype, 
		fitnesses: Fitness, 
		descriptors: Descriptor, 
		centroids: Centroid, 
		extra_scores: ExtraScores|None = None,
		sigma: float=0.1) :

		genotype = jax.tree.map(lambda x: x[0], genotypes)
		repertoire = cls.init_default(genotype, centroids, sigma)
		new_repertoire = repertoire.add(genotypes, descriptors, fitnesses, extra_scores)
		return new_repertoire

	@classmethod
	def init_default(cls, 
		genotype: Genotype, 
		centroids: Centroid, 
		sigma: float=0.1):

		num_cells = centroids.shape[0]
		default_fitnesses = jnp.full(num_cells, -jnp.inf)
		default_genotypes = jax.tree.map(lambda x: jnp.zeros((num_cells, *x.shape)), genotype)
		default_descriptors = jnp.zeros_like(centroids)
		default_scores = jnp.full(num_cells, -jnp.inf)
		return cls(default_genotypes, default_fitnesses, default_descriptors, centroids, default_scores, sigma)




if __name__ == '__main__':
	from qdax.core.containers.mapelites_repertoire import compute_euclidean_centroids
	genotypes = jr.uniform(jr.key(1), (100,2))
	fitness = jr.uniform(jr.key(2), (100,3))
	bds = jr.uniform(jr.key(3), (100,3,2))
	centroids = compute_euclidean_centroids((32,32), 0, 1)

	repertoire = IlluminatePotentialRepertoire.init(genotypes, fitness, bds, centroids)


			