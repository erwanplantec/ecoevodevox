"""MAP-Elites repertoire.

Copied from the EcoEvoDevoNoise project (`src/qd/mapelites.py`). The two helpers it
relied on there (`do_hierarchical_selection`, `pytree_repeat`) are inlined below so
this module is self-contained.

On top of a vanilla MAP-Elites grid this repertoire also tracks, per niche, who
discovered it and when (`NicheInfos`), and per elite its age/generation
(`GenotypeInfos`). Elites can be aged out (`max_age`), randomly replaced
(`replace_probability`) or randomly wiped (`extinction_probability`) — set those to
their defaults to recover standard MAP-Elites.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
from jaxtyping import PyTree, UInt32
from flax.struct import PyTreeNode

from qdax.core.containers.repertoire import Repertoire
from qdax.core.containers.mapelites_repertoire import get_cells_indices


def pytree_repeat(pytree, k: int, axis=0, unsqueeze: bool=True):
    # ---
    if unsqueeze:
        _fn = lambda x: jnp.expand_dims(x, axis=axis)
    else:
        _fn = lambda x: x
    fn = lambda x: jnp.repeat(_fn(x), k, axis=axis)
    # ---
    return jax.tree.map(fn, pytree)


def do_hierarchical_selection(selection_mask, selection_vars, indices, num_centroids):
    """
    Args:
        selection_mask: pre computed slection masl
        selections_vars: variables of comparisom
        indices: indices of niches ind. compete for
        num_centroids: total number of niches
    """
    def _keep_best_of_var(select_var):
        # keep best fitnesses
        max_vals = jax.ops.segment_max(
            select_var,
            indices,
            num_segments=num_centroids,
        )
        cond_values = jnp.take_along_axis(max_vals, indices, 0)
        is_selected = select_var == cond_values
        return is_selected

    def _count_candidates_per_niche(selection_mask, indices):
        nb_candidates = jax.ops.segment_sum(selection_mask.astype(int), indices.astype(int), num_centroids)
        return nb_candidates

    def _selection_step(carry):
        selection_mask, var_id = carry
        selection_var = jnp.where(selection_mask, selection_vars[var_id], -jnp.inf)
        var_selection_mask = _keep_best_of_var(selection_var)
        selection_mask = selection_mask & var_selection_mask
        return selection_mask, var_id+1

    def _cond_fn(carry):
        selection_mask, var_id = carry
        return (var_id<selection_vars.shape[0])&(_count_candidates_per_niche(selection_mask, indices).max()>1)


    selection_mask, *_ = jax.lax.while_loop(
        cond_fun=_cond_fn, body_fun=_selection_step, init_val=(selection_mask, 0)
    )

    return selection_mask


class GenotypeInfos(PyTreeNode):
    # ---
    age: jax.Array
    generation: jax.Array
    # ---
    @classmethod
    def default(cls):
        return GenotypeInfos(generation=jnp.zeros((), dtype=jnp.uint32), age=jnp.zeros((), dtype=jnp.uint32))
    # ---

class ParentData(PyTreeNode):
    # ---
    niche_ids: jax.Array
    fitnesses: jax.Array
    infos: GenotypeInfos
    # ---
    @classmethod
    def default(cls, genotype_like: PyTree):
        return cls(niche_ids=jnp.full((), jnp.iinfo(jnp.uint32).max, dtype=jnp.uint32),
                   fitnesses=jnp.full((), -jnp.inf),
                   infos=GenotypeInfos.default())


class NicheInfos(PyTreeNode):
    # ---
    visit_count: jax.Array
    discovery_step: jax.Array
    discovery_from: jax.Array
    last_addition_step: jax.Array
    last_addition_from: jax.Array
    # ---
    @classmethod
    def default(cls):
        return cls(discovery_step=jnp.asarray(jnp.iinfo(jnp.uint32).max, dtype=jnp.uint32),
                   discovery_from=jnp.asarray(jnp.iinfo(jnp.uint32).max, dtype=jnp.uint32),
                   last_addition_step=jnp.asarray(jnp.iinfo(jnp.uint32).max, dtype=jnp.uint32),
                   last_addition_from=jnp.asarray(jnp.iinfo(jnp.uint32).max, dtype=jnp.uint32),
                   visit_count=jnp.zeros((), dtype=jnp.uint32))

class UpdateInfos(PyTreeNode):
    # ----
    added: jax.Array # mask of added genotypes in current gen (N,)
    discovered_new_niches: jax.Array # mask of genotyoes having disc new niche (N,)
    added_where: jax.Array
    # ---

class MapElitesRepertoire(Repertoire):
    # ---
    genotypes: PyTree
    fitnesses: jax.Array
    descriptors: jax.Array
    centroids: jax.Array
    niche_ids: jax.Array
    task_extra_scores: PyTree
    niche_infos: NicheInfos
    parents_infos: ParentData
    genotype_infos: GenotypeInfos
    step: UInt32
    # ---
    replace_probability: float
    max_age: int|float
    extinction_probability: float
    # ---
    @classmethod
    def extract_parents_data(cls, repertoire: PyTree)->ParentData:
        data =  ParentData(
            niche_ids=repertoire.niche_ids,
            fitnesses=repertoire.fitnesses,
            infos=repertoire.genotype_infos
        )
        return data
    # ---
    @property
    def num_centroids(self):
        return self.centroids.shape[0]
    @property
    def num_descriptors(self):
        return self.descriptors.shape[2]
    @property
    def B(self):
        return self.num_descriptors
    def is_occupied(self):
        return ~jnp.isinf(self.fitnesses)
    # ---
    def select(self, key: jax.Array, num_samples: int, selector=None):
        num_centroids = self.centroids.shape[0]
        occupied = self.is_occupied()
        p = occupied / occupied.sum()
        indices = jr.choice(key, num_centroids, (num_samples,), p=p)
        return jax.tree.map(lambda x: x[indices] if x.shape else x, self)
    # ---
    def add(self, #type:ignore
            batch_of_genotypes: PyTree,
            batch_of_fitnesses: jax.Array,
            batch_of_descriptors: jax.Array,
            batch_of_task_extra_scores: PyTree,
            batch_of_parents_infos: ParentData,
            key: jax.Array):

        if batch_of_fitnesses.ndim==2:
            N, K, _ = batch_of_descriptors.shape
            batch_of_genotypes, batch_of_parents_infos = jax.tree.map(
                lambda x: x[:,None].repeat(K, axis=1).reshape(N*K, *x.shape[1:]),
                [batch_of_genotypes, batch_of_parents_infos]
            )
            batch_of_fitnesses, batch_of_descriptors, batch_of_task_extra_scores = jax.tree.map(
                lambda x: x.reshape(N*K, *x.shape[2:]),
                [batch_of_fitnesses, batch_of_descriptors, batch_of_task_extra_scores]
            )

        k1, k2, k3 = jr.split(key, 3)

        batch_of_indices = get_cells_indices(batch_of_descriptors, self.centroids)

        num_centroids = self.centroids.shape[0]

        fitness_mask = jr.bernoulli(k1, self.replace_probability, (num_centroids,)) & (self.genotype_infos.age>self.max_age)
        modified_fitnesses = jnp.where(fitness_mask, -jnp.inf,  self.fitnesses)
        current_fitnesses = modified_fitnesses[batch_of_indices]

        addition_condition = batch_of_fitnesses >= current_fitnesses

        addition_condition = do_hierarchical_selection(addition_condition,
                                                       jnp.stack([batch_of_fitnesses,
                                                                  jr.uniform(k2, batch_of_fitnesses.shape)], axis=0),
                                                       batch_of_indices,
                                                       num_centroids)

        # assign fake position when relevant : num_centroids is out of bound
        batch_of_indices = jnp.where(
            addition_condition, batch_of_indices, num_centroids
        )

        # create new repertoire
        new_repertoire_genotypes = jax.tree.map(
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

        # update extra scores
        new_task_extra_scores = jax.tree.map(
            lambda repertoire_scores, new_scores: repertoire_scores.at[
                batch_of_indices
            ].set(new_scores),
            self.task_extra_scores,
            batch_of_task_extra_scores if batch_of_task_extra_scores is not None else {},
        )

        discovered_niches = jnp.isinf(self.fitnesses) & ~jnp.isinf(new_fitnesses)
        updated_from = jnp.zeros_like(self.niche_ids).at[batch_of_indices].set(batch_of_parents_infos.niche_ids)
        is_updated = jnp.zeros(num_centroids, dtype=jnp.bool).at[batch_of_indices].set(True)

        niche_infos = NicheInfos(
            visit_count=self.niche_infos.visit_count+is_updated.astype(jnp.uint32),
            discovery_step=jnp.where(discovered_niches, self.step, self.niche_infos.discovery_step), #type:ignore
            discovery_from=jnp.where(discovered_niches, updated_from, self.niche_infos.discovery_from),
            last_addition_step=jnp.where(is_updated, self.step, self.niche_infos.last_addition_step), #type:ignore
            last_addition_from=jnp.where(is_updated, updated_from, self.niche_infos.last_addition_from),
        )

        new_parents_infos = jax.tree.map(lambda a, b: a.at[batch_of_indices].set(b),
                                         self.parents_infos,
                                         batch_of_parents_infos)

        new_ages = jnp.where(~jnp.isinf(new_fitnesses), (self.genotype_infos.age+1).at[batch_of_indices].set(1), 0)
        new_fitnesses = jnp.where(self.genotype_infos.age>self.max_age, -jnp.inf, new_fitnesses)
        new_fitnesses = jnp.where(jr.bernoulli(k3, self.extinction_probability, new_fitnesses.shape), -jnp.inf, new_fitnesses)

        new_genotype_infos = GenotypeInfos(generation=self.genotype_infos.generation.at[batch_of_indices].set(batch_of_parents_infos.infos.generation+1),
                                           age=new_ages)

        new_repertoire = self.replace(
            genotypes=new_repertoire_genotypes,
            fitnesses=new_fitnesses,
            descriptors=new_descriptors,
            task_extra_scores=new_task_extra_scores,
            parents_infos=new_parents_infos,
            niche_infos=niche_infos,
            step=self.step+1,
            genotype_infos=new_genotype_infos
        )

        discovered_new_niches = addition_condition & jnp.isinf(self.fitnesses)[batch_of_indices]
        update_infos = UpdateInfos(added=addition_condition,
                                   discovered_new_niches=discovered_new_niches,
                                   added_where=batch_of_indices)

        return new_repertoire, update_infos
    # ---
    @classmethod
    def init(cls, #type:ignore
             batch_of_genotypes,
             batch_of_fitnesses,
             batch_of_descriptors,
             centroids,
             batch_of_task_extra_scores,
             key,
             **kwargs):

        task_extra_scores = {} if batch_of_task_extra_scores is None else batch_of_task_extra_scores
        N, *_, B = batch_of_descriptors.shape

        take_one = lambda tree: jax.tree.map(lambda x: x[0], tree)

        repertoire = cls.init_default(genotype=take_one(batch_of_genotypes),
                                      centroids=centroids,
                                      task_extra_score=take_one(take_one(task_extra_scores)),
                                      **kwargs)

        repertoire, _ = repertoire.add(batch_of_genotypes=batch_of_genotypes,
                                       batch_of_fitnesses=batch_of_fitnesses,
                                       batch_of_descriptors=batch_of_descriptors,
                                       batch_of_task_extra_scores=task_extra_scores,
                                       batch_of_parents_infos=pytree_repeat(ParentData.default(take_one(batch_of_genotypes)), N),
                                       key=key)

        return repertoire

    @classmethod
    def init_default(cls,
                     genotype,
                     centroids,
                     task_extra_score,
                     max_age: int|float=1e10,
                     replace_probability: float=0.0,
                     extinction_probability: float=0.0,
                     **kwargs):
        num_centroids, B = centroids.shape

        # default fitness is -inf
        default_fitnesses = -jnp.inf * jnp.ones(shape=num_centroids)

        # default genotypes is all 0
        default_genotypes = jax.tree_util.tree_map(
            lambda x: jnp.zeros(shape=(num_centroids,) + x.shape, dtype=x.dtype),
            genotype,
        )

        # default descriptor is all zeros
        default_descriptors = jnp.zeros_like(centroids)

        default_niche_infos = pytree_repeat(NicheInfos.default(), num_centroids)

        default_genotype_infos = pytree_repeat(GenotypeInfos.default(), num_centroids)

        default_parent_infos = pytree_repeat(ParentData.default(genotype), num_centroids)


        default_task_extra_scores = jax.tree.map(
            lambda x: jnp.zeros((num_centroids, *x.shape)),
            task_extra_score
        )

        step = jnp.zeros((), dtype=jnp.uint32)

        ids = jnp.arange(num_centroids, dtype=jnp.uint32)

        return cls(
            genotypes=default_genotypes,
            fitnesses=default_fitnesses,
            descriptors=default_descriptors,
            centroids=centroids,
            task_extra_scores=default_task_extra_scores,
            parents_infos=default_parent_infos,
            niche_infos=default_niche_infos,
            step=step,
            niche_ids=ids,
            genotype_infos=default_genotype_infos,
            max_age=max_age,
            replace_probability=replace_probability,
            extinction_probability=extinction_probability,
            **kwargs
        )



class FitnessMapElites(MapElitesRepertoire):
    """
    MapElites with fitness-proportional parent selection.

    Identical to MapElitesRepertoire in every respect except that `select`
    samples parents with probability proportional to exp(fitness) (softmax)
    rather than uniformly from occupied niches.  Unoccupied niches have
    fitness -inf and therefore receive probability zero automatically.
    """

    def select(self, key: jax.Array, num_samples: int, selector=None):
        p = jnn.softmax(self.fitnesses)   # -inf → 0; occupied niches weighted by fitness
        indices = jr.choice(key, self.centroids.shape[0], (num_samples,), p=p)
        return jax.tree.map(lambda x: x[indices] if x.shape else x, self)


class GeneticAlgorithmRepertoire(MapElitesRepertoire):
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def select(self, key: jax.Array, num_samples: int, selector=None):
        return super().select(key, num_samples, selector)
    # ------------------------------------------------------------------
    def add(self, batch_of_genotypes: PyTree, batch_of_fitnesses: jax.Array, batch_of_descriptors: jax.Array, batch_of_task_extra_scores: PyTree, batch_of_parents_infos: ParentData, key: jax.Array): #type:ignore
        if batch_of_fitnesses.ndim==2:
            N, K, _ = batch_of_descriptors.shape
            batch_of_genotypes, batch_of_parents_infos = jax.tree.map(
                lambda x: x[:,None].repeat(K, axis=1).reshape(N*K, *x.shape[1:]),
                [batch_of_genotypes, batch_of_parents_infos]
            )
            batch_of_fitnesses, batch_of_descriptors, batch_of_task_extra_scores = jax.tree.map(
                lambda x: x.reshape(N*K, *x.shape[2:]),
                [batch_of_fitnesses, batch_of_descriptors, batch_of_task_extra_scores]
            )
        else:
            N, K = len(batch_of_fitnesses), 1

        all_fitnesses, all_descriptors,  all_genotypes, all_parent_infos, all_task_extra_scores = jax.tree.map(
            lambda a, b: jnp.concat([a,b], axis=0),
            [batch_of_fitnesses, batch_of_descriptors, batch_of_genotypes, batch_of_parents_infos, batch_of_task_extra_scores],
            [self.fitnesses, self.descriptors, self.genotypes, self.parents_infos, self.task_extra_scores]
        )

        ages = jnp.concatenate([jnp.zeros(batch_of_fitnesses.shape, dtype=self.genotype_infos.age.dtype), self.genotype_infos.age], axis=0)
        all_fitnesses = jnp.where(ages>self.max_age, -jnp.inf, all_fitnesses); assert isinstance(all_fitnesses, jax.Array)

        best_ids = jnp.argsort(all_fitnesses, descending=True)[:len(self.fitnesses)]

        new_fitnesses, new_descriptors,  new_genotypes, new_parent_infos, new_task_extra_scores, new_ages = jax.tree.map(
            lambda x: x[best_ids],
            [all_fitnesses, all_descriptors,  all_genotypes, all_parent_infos, all_task_extra_scores, ages+1] 
        )

        all_generations = jnp.concatenate([batch_of_parents_infos.infos.generation+1, self.genotype_infos.generation])
        new_generations = all_generations[best_ids]

        new_genotype_infos = GenotypeInfos(age=new_ages, generation=new_generations)

        repertoire = GeneticAlgorithmRepertoire(fitnesses=new_fitnesses, 
                                                descriptors=new_descriptors,
                                                task_extra_scores=new_task_extra_scores,
                                                genotypes=new_genotypes,
                                                parents_infos=new_parent_infos,
                                                genotype_infos=new_genotype_infos,
                                                max_age=self.max_age,
                                                replace_probability=self.replace_probability,
                                                extinction_probability=self.extinction_probability,
                                                centroids=self.centroids,
                                                niche_ids=self.niche_ids,
                                                niche_infos=self.niche_infos,
                                                step=self.step+1)

        update_infos = UpdateInfos(added=jnp.zeros(N*K,), discovered_new_niches=jnp.zeros(N*K,), added_where=jnp.zeros(N*K,))

        return repertoire, update_infos


    # ------------------------------------------------------------------