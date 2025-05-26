import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import equinox as eqx
import equinox.nn as nn
from flax.struct import PyTreeNode
from jax.flatten_util import ravel_pytree

from .base import DevelopmentalModel
from src.agents.nn.ctrnn import SECTRNN
from src.agents.nn.rnn import SERNN

def heaviside(x: jax.Array)->jax.Array:
    return jnp.where(x>0, 1., 0.)

def M(x):
    return jnp.stack([x[...,0], x[...,1], jnp.abs(x[...,0]), jnp.abs(x[...,1]), jnp.maximum(jnp.abs(x[...,0]), jnp.abs(x[...,1]))], axis=-1)

class RANDState(PyTreeNode):
    S: jax.Array
    X: jax.Array
    mask: jax.Array
    mitotic_factors: jax.Array
    death_factors: jax.Array

class RNNState(SERNN):
	s: jax.Array
	m: jax.Array

class CTRNNState(SECTRNN):
	s: jax.Array
	m: jax.Array

type Network=RNNState|CTRNNState

class RAND(DevelopmentalModel):
    # --- GRN prms
    W_in: nn.Linear
    W_ex: nn.Linear
    W_migr: jax.Array
    W_pert: jax.Array
    W_syn: jax.Array
    bias: jax.Array
    tau: jax.Array
    O_syn: jax.Array
    # ---
    nb_neurons: int
    max_neurons: int
    genes_shaper: callable
    total_genes: int
    max_mitosis: int
    mitotic_factor_threshold: float
    death_factor_threshold: float
    communication_fields: int
    grn_model: str
    autonomous_decay: bool
    dev_iters: int
    network_type: str
    # ---
    def __init__(self, nb_neurons=1, max_neurons=128, regulatory_genes=8, migratory_genes=4, perturbation_genes=2, sensory_genes=1, motor_genes=1,
                 synaptic_genes=4, synaptic_proteins=4, max_mitosis=10, mitotic_factor_threshold=10.0, death_factor_threshold=10.0, 
                 communication_fields=8, nb_synaptic_rules=1, grn_model="continuous", autonomous_decay=True, dev_iters=400,
                 network_type="ctrnn", *, key: jax.Array):
        """Initialize the RAND (Regulation based Neural Development) model.
        
        Args:
            nb_neurons (int): Initial number of neurons. Defaults to 1.
            max_neurons (int): Maximum number of neurons allowed. Defaults to 128.
            regulatory_genes (int): Number of regulatory genes. Defaults to 8.
            migratory_genes (int): Number of migratory genes. Defaults to 4.
            perturbation_genes (int): Number of perturbation genes. Defaults to 2.
            sensory_genes (int): Number of sensory genes. Defaults to 1.
            motor_genes (int): Number of motor genes. Defaults to 1.
            synaptic_genes (int): Number of synaptic genes. Defaults to 4.
            synaptic_proteins (int): Number of synaptic proteins. Defaults to 4.
            max_mitosis (int): Maximum number of mitosis events per step. Defaults to 10.
            mitotic_factor_threshold (float): Threshold for mitosis activation. Defaults to 10.0.
            death_factor_threshold (float): Threshold for cell death. Defaults to 10.0.
            communication_fields (int): Number of communication fields. Defaults to 8.
            nb_synaptic_rules (int): Number of synaptic rules. Defaults to 1.
            grn_model (str): Type of gene regulatory network model. Defaults to "continuous".
            autonomous_decay (bool): Whether to apply autonomous decay. Defaults to True.
            dev_iters (int): Number of development iterations. Defaults to 400.
            network_type (str): Type of neural network ("ctrnn" or "rnn"). Defaults to "ctrnn".
            key (jax.Array): JAX random key for initialization.
        """
        self.nb_neurons = nb_neurons

        genome_compartments = [
            jnp.zeros(1), # speed
            jnp.zeros(migratory_genes), # migr fields
            jnp.zeros(perturbation_genes), # pert fields
            jnp.zeros(synaptic_genes), # syn fields
            jnp.zeros(1), #mitosis
            jnp.zeros(1), #death
            jnp.zeros(regulatory_genes), # regulatory
            jnp.zeros(sensory_genes), # sensory
            jnp.zeros(motor_genes) # motor
        ]

        if network_type=="ctrnn":
            genome_compartments.append(
                (jnp.zeros(()), jnp.zeros(()), jnp.zeros(()))
            )
        elif network_type=="rnn":
            genome_compartments.append(jnp.zeros(()))

        genome, self.genes_shaper = ravel_pytree(genome_compartments)
        total_genes = len(genome)
        self.total_genes = total_genes
        
        kin, kex, kmigr, kpert, kbias, ksyn, kosyns, ktau = jr.split(key, 8)
        
        scale = jnp.sqrt(1/total_genes)
        self.W_in = nn.Linear(total_genes, total_genes, key=kin, use_bias=False)
        self.W_ex = nn.Linear(5+communication_fields, total_genes, key=kex, use_bias=True)
        self.bias = jr.normal(kbias, (total_genes,))*scale
        self.tau = jr.uniform(ktau, (total_genes,))
        
        self.W_migr = jr.normal(kmigr, (migratory_genes,communication_fields+5))*0.1
        self.W_pert = jr.normal(kpert, (perturbation_genes,communication_fields))*0.1
        
        self.W_syn = jr.normal(ksyn, (synaptic_genes,synaptic_proteins))*0.1
        self.O_syn = jr.normal(kosyns, (nb_synaptic_rules,synaptic_proteins,synaptic_proteins))*0.1
        self.max_mitosis = max_mitosis
        self.max_neurons = max_neurons
        self.mitotic_factor_threshold = mitotic_factor_threshold
        self.death_factor_threshold = death_factor_threshold
        self.communication_fields = communication_fields
        self.grn_model = grn_model
        self.autonomous_decay = autonomous_decay
        self.dev_iters = dev_iters
        self.network_type = network_type
    # ---
    def init(self, key: jax.Array)->RANDState:
        """Initialize the state of the RAND model.
        
        Args:
            key (jax.Array): JAX random key for initialization.
            
        Returns:
            RANDState: Initial state containing gene expression levels (S), positions (X),
                      neuron mask, mitotic factors, and death factors.
        """
        S = jnp.zeros((self.max_neurons, self.total_genes))
        if self.nb_neurons==1:
            X = jnp.zeros((self.max_neurons, 2))
        else:
            X = jr.normal(key, (self.max_neurons, 2))*0.01
        mask = jnp.arange(self.max_neurons) < self.nb_neurons
        mitotic_factors = jnp.zeros((self.max_neurons,))
        death_factors = jnp.zeros((self.max_neurons,))
        return RANDState(S, X, mask, mitotic_factors, death_factors)
    # ---
    def step(self, state: RANDState, key: jax.Array)->RANDState:
        """Perform one step of development in the RAND model.
        
        This method handles:
        1. Mitosis and death of neurons
        2. Gene regulatory network updates
        3. Neuron migration
        4. State updates
        
        Args:
            state (RANDState): Current state of the system.
            key (jax.Array): JAX random key for stochastic operations.
            
        Returns:
            tuple: (RANDState, dict) containing:
                - Updated state
                - Dictionary with additional information about the step
        """
         # --- Mitosis and death step
        _, _, _, _, S_mitosis, S_death, *_ = jax.vmap(self.genes_shaper)(state.S)
        death_gene_active = (S_death>0.9).squeeze(-1)     & state.mask
        mitotic_gene_active = (S_mitosis>0.9).squeeze(-1) & state.mask
        mitotic_factors = jnp.clip(
            jnp.where(mitotic_gene_active, state.mitotic_factors+1, 0), 0, self.mitotic_factor_threshold+1
        )
        death_factors = jnp.clip(
            jnp.where(death_gene_active, state.death_factors+1, 0), 0, self.death_factor_threshold+1
        )
        state = state.replace(mitotic_factors=mitotic_factors, death_factors=death_factors)

        dead = death_factors > self.death_factor_threshold
        death_factors = jnp.where(dead, 0., death_factors)
        mask = jnp.where(dead, False, state.mask)
        state = state.replace(mask=mask)

        def _mitosis(state, mitotic, key):
            k1, k2 = jr.split(key)
            _, mitotic_ids = jax.lax.top_k(mitotic+jr.normal(k1, (self.max_neurons,))*0.01, self.max_mitosis)
            is_mitotic = mitotic[mitotic_ids]
            is_free, buffer_ids = jax.lax.top_k(~state.mask, self.max_mitosis)
            mitosis_mask = is_mitotic & is_free
            
            buffer_ids = jnp.where(mitosis_mask, buffer_ids, self.nb_neurons+1)
            mitotic_ids = jnp.where(mitosis_mask, mitotic_ids, self.nb_neurons+1)

            mask = state.mask.at[buffer_ids].set(mitosis_mask)
            S = state.S.at[buffer_ids].set(state.S[mitotic_ids])
            X = state.X.at[buffer_ids].set(state.X[mitotic_ids] + jr.normal(k2, (self.max_mitosis,2))*0.01)
            mitotic_factors = state.mitotic_factors.at[mitotic_ids].set(0)
            mitotic_factors = mitotic_factors.at[buffer_ids].set(0)
        
            return state.replace(S=S, X=X, mask=mask, mitotic_factors=mitotic_factors), mitosis_mask.sum()
        
        has_free_space = jnp.any(~state.mask)
        mitotic = (mitotic_factors > self.mitotic_factor_threshold) & state.mask
        state, nb_mitotic = jax.lax.cond(has_free_space & jnp.any(mitotic), 
                             _mitosis, 
                             lambda s, *_: (s, 0), 
                             state, mitotic, key)
        
        _, _, S_pert, *_ = jax.vmap(self.genes_shaper)(state.S) #N, m
        zetas = S_pert@self.W_pert
        def M_(x):
            dists = jnp.square(x[None]-state.X).sum(-1) # N,
            concentrations = jnp.exp(-dists/0.005) # N, 
            perts = jnp.sum(concentrations[:,None] * zetas * state.mask[:,None], axis=0)
            return jnp.concatenate([M(x),perts])
        # --- GRN step
        I = jnn.sigmoid(jax.vmap(M_)(state.X))
        inp = jax.vmap(self.W_ex)(I)
        if self.grn_model == "continuous":
            dS = jnn.tanh(jax.vmap(self.W_in)(state.S) + inp + self.bias)
            if self.autonomous_decay:
                dS = dS - state.S
            dS = dS / self.tau[None]
            S = jnp.clip(state.S + 0.03 * dS, -1.0, 1.0)
            S = jnp.where(state.mask[:,None], S, 0.)
        else:
            S = heaviside(jax.vmap(self.W_in)(state.S) + inp + self.bias)
        # --- Migration step
        S_mvt, S_migr, S_pert, *_ = jax.vmap(self.genes_shaper)(state.S)
        
        lambda_ = S_migr@self.W_migr
        
        @jax.grad
        def energy_fn(x, lambda_):
            return jnp.sum(lambda_*M_(x), axis=-1)
        
        vel = jnn.sigmoid(S_mvt*5)
        dX = jax.vmap(energy_fn)(state.X, lambda_)
        dX = jnp.clip(dX, -1., 1.)
        
        X = jnp.clip(state.X + 0.05*dX*vel, -1., 1.)
        X = jnp.where(state.mask[:,None], X, 0.)

        state = state.replace(X=X, S=S)

        return state, {"state": state, "nb_mitotic": nb_mitotic, "mitotic": mitotic, "population": state.mask.sum()}
    # --
    def make_network(self, state: RANDState)->Network:
        """Create a neural network from the current developmental state.
        
        This method extracts synaptic weights, sensory and motor neurons,
        and creates either a CTRNN or RNN based on the network_type.
        
        Args:
            state (RANDState): Current developmental state.
            
        Returns:
            Network: Either a CTRNNState or RNNState containing the neural network
                    parameters and structure.
        """
        _, _, _, S_syn, *_, S_sensory, S_motor, S_neurons = jax.vmap(self.genes_shaper)(state.S)
        get_weigth = lambda O: S_syn@O@S_syn.T
        W = jax.vmap(get_weigth)(self.O_syn).sum(0)
        W = jnp.where(state.mask[None]*state.mask[:,None], W, 0.)
        W = jnp.where(jnp.abs(W)>1e-3, W, 0.)
        sensory = jnn.sigmoid(S_sensory*5.0)
        motor = jnn.sigmoid(S_motor*5.0)
        v = jnp.zeros((self.max_neurons,))
        mask = state.mask
        X = state.X
        if self.network_type=="ctrnn":
            tau, gain, bias = S_neurons
            return CTRNNState(v=v, W=W, mask=mask, x=X, s=sensory, m=motor, tau=tau, gain=gain, b=bias)
        elif self.network_type=="rnn":
            bias = S_neurons
            return RNNState(v=v, W=W, mask=mask, x=X, s=sensory, m=motor, b=bias)
    # ---
    def __call__(self, key: jax.Array)->Network:
        """Run the complete development process and return the resulting network.
        
        This method:
        1. Initializes the state
        2. Runs the development process for dev_iters steps
        3. Creates and returns the final neural network
        
        Args:
            key (jax.Array): JAX random key for the development process.
            
        Returns:
            Network: The final neural network after development.
        """
        key_init, key_scan = jr.split(key)
        state = self.init(key_init)
        state = jax.lax.scan(lambda s, k: self.step(s, k), state, jr.split(key_scan, self.dev_iters))[0]
        return self.make_network(state)
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    mdl = RAND(nb_neurons=1, network_type="rnn", key=jr.key(2))
    net = mdl(jr.key(2))
    plt.scatter(*net.x.T)
    plt.show()

