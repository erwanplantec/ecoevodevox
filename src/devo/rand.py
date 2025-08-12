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
    apoptosis_factors: jax.Array

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
    W_sign: jax.Array
    W_syn: jax.Array
    W_neur: nn.Linear|None
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
    apoptosis_factor_threshold: float
    signalling_proteins: int
    grn_model: str
    autonomous_decay: bool
    dev_iters: int
    network_type: str
    expression_bounds: tuple[float, float]
    motor_activation_fn: callable
    sensory_activation_fn: callable
    gene_noise_scale: float
    position_noise_scale: float
    # ---
    def __init__(self, nb_neurons=1, max_neurons=128, regulatory_genes=8, migratory_genes=4, signalling_genes=2, sensory_genes=1, motor_genes=1,
                 synaptic_genes=4, synaptic_proteins=4, signalling_proteins=4, max_mitosis=10, mitotic_factor_threshold=10.0, apoptosis_factor_threshold=10.0, 
                 nb_synaptic_rules=1, grn_model="continuous", autonomous_decay=True, dev_iters=400, gene_noise_scale=0.0, position_noise_scale=0.0,
                 neuron_params_genes=1, network_type="ctrnn", expression_bounds=(0.0, 1.0), motor_activation_fn=lambda x:x, sensory_activation_fn=lambda x:x, *, key: jax.Array):
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
            apoptosis_factor_threshold (float): Threshold for cell death. Defaults to 10.0.
            communication_fields (int): Number of communication fields. Defaults to 8.
            nb_synaptic_rules (int): Number of synaptic rules. Defaults to 1.
            grn_model (str): Type of gene regulatory network model. Defaults to "continuous".
            autonomous_decay (bool): Whether to apply autonomous decay. Defaults to True.
            dev_iters (int): Number of development iterations. Defaults to 400.
            network_type (str): Type of neural network ("ctrnn" or "rnn"). Defaults to "ctrnn".
            key (jax.Array): JAX random key for initialization.
        """
        self.nb_neurons = nb_neurons

        genome_compartments = {
            "speed": jnp.zeros(1), # speed
            "migratory": jnp.zeros(migratory_genes), # migr fields
            "signalling": jnp.zeros(signalling_genes), # pert fields
            "synaptic": jnp.zeros(synaptic_genes), # syn fields
            "mitosis": jnp.zeros(1), #mitosis
            "apoptosis": jnp.zeros(1), #apoptosis
            "regulatory": jnp.zeros(regulatory_genes), # regulatory
            "sensory": jnp.zeros(sensory_genes), # sensory
            "motor": jnp.zeros(motor_genes), # motor
            "neuron_params": jnp.zeros(neuron_params_genes)
        }

        genome, self.genes_shaper = ravel_pytree(genome_compartments)
        total_genes = len(genome)
        self.total_genes = total_genes
        
        kin, kex, kmigr, kpert, kbias, ksyn, kosyns, ktau, kneur = jr.split(key, 9)

        if network_type=="ctrnn":
            self.W_neur = nn.Linear(neuron_params_genes, 3, key=kneur)
        elif network_type=="rnn":
            self.W_neur = nn.Linear(neuron_params_genes, "scalar", key=kneur)
        
        self.W_in = jr.uniform(kin, (total_genes, total_genes), minval=-1.0, maxval=1.0)
        self.W_ex = jr.uniform(kex, (5+signalling_proteins, total_genes), minval=-1.0, maxval=1.0)
        self.bias = jr.uniform(kbias, (total_genes,), minval=-1.0, maxval=1.0)
        self.tau = jr.uniform(ktau, (total_genes,), minval=0.1, maxval=1.0)
        
        self.W_migr = jr.normal(kmigr, (migratory_genes,signalling_proteins+5))
        self.W_sign = jr.normal(kpert, (signalling_genes,signalling_proteins))
        
        self.W_syn = jr.normal(ksyn, (synaptic_genes,synaptic_proteins))
        self.O_syn = jr.normal(kosyns, (nb_synaptic_rules,synaptic_proteins,synaptic_proteins))
        self.max_mitosis = max_mitosis
        self.max_neurons = max_neurons
        self.mitotic_factor_threshold = mitotic_factor_threshold
        self.apoptosis_factor_threshold = apoptosis_factor_threshold
        self.signalling_proteins = signalling_proteins
        self.grn_model = grn_model
        self.autonomous_decay = autonomous_decay
        self.dev_iters = dev_iters
        self.network_type = network_type
        self.expression_bounds = expression_bounds
        self.motor_activation_fn = motor_activation_fn
        self.sensory_activation_fn = sensory_activation_fn
        self.gene_noise_scale = gene_noise_scale
        self.position_noise_scale = position_noise_scale
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
        apoptosis_factors = jnp.zeros((self.max_neurons,))
        return RANDState(S, X, mask, mitotic_factors, apoptosis_factors)
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
        key_mitosis, key_gene, key_position = jr.split(key, 3)
         # --- Mitosis and death step
        genes = jax.vmap(self.genes_shaper)(state.S)
        apoptosis_gene_active = (genes["apoptosis"]>0.9).squeeze(-1) & state.mask
        mitotic_gene_active = (genes["mitosis"]>0.9).squeeze(-1) & state.mask
        mitotic_factors = jnp.clip(
            jnp.where(mitotic_gene_active, state.mitotic_factors+1, 0), 0, self.mitotic_factor_threshold+1
        )
        apoptosis_factors = jnp.clip(
            jnp.where(apoptosis_gene_active, state.apoptosis_factors+1, 0), 0, self.apoptosis_factor_threshold+1
        )
        state = state.replace(mitotic_factors=mitotic_factors, apoptosis_factors=apoptosis_factors)

        dead = apoptosis_factors > self.apoptosis_factor_threshold
        apoptosis_factors = jnp.where(dead, 0., apoptosis_factors)
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
                             state, mitotic, key_mitosis)
        
        genes = jax.vmap(self.genes_shaper)(state.S) #N, m
        signals = genes["signalling"]@self.W_sign
        def M_(x):
            dists = jnp.square(x[None]-state.X).sum(-1) # N,
            concentrations = jnp.exp(-dists/0.005) # N, 
            perts = jnp.sum(concentrations[:,None] * signals * state.mask[:,None], axis=0)
            return jnp.concatenate([M(x),perts])
        # --- GRN step
        I = jax.vmap(M_)(state.X)
        inp_ex = I@self.W_ex
        inp_in = state.S@self.W_in

        if self.grn_model == "continuous":
            dS = jnn.tanh(inp_in + inp_ex + self.bias) + (jr.normal(key_gene, state.S.shape)*self.gene_noise_scale)
            if self.autonomous_decay:
                dS = dS - state.S
            dS = dS / self.tau[None]
            S = jnp.clip(state.S + 0.03 * dS, self.expression_bounds[0], self.expression_bounds[1])
            S = jnp.where(state.mask[:,None], S, 0.)
        else:
            S = heaviside(inp_in + inp_ex + self.bias)
        # --- Migration step
        genes = jax.vmap(self.genes_shaper)(state.S)
        S_mvt, S_migr = genes["speed"], genes["migratory"]
        
        lambda_ = S_migr@self.W_migr
        
        @jax.grad
        def energy_fn(x, lambda_):
            return jnp.sum(lambda_*M_(x), axis=-1)
        
        vel = S_mvt
        dX = -jax.vmap(energy_fn)(state.X, lambda_)
        dX = jnp.clip(dX, -1., 1.) + (jr.normal(key_position, dX.shape)*self.position_noise_scale)
        
        X = jnp.clip(state.X + 0.05*dX*vel, -1., 1.)
        X = jnp.where(state.mask[:,None], X, 0.)

        state = state.replace(X=X, S=S)

        return state, {"state": state, "nb_mitotic": nb_mitotic, "mitotic": mitotic, "population": state.mask.sum()}
    # ---
    def do_migration(self, state: RANDState, key: jax.Array)->tuple[RANDState,RANDState]:
        state, states = jax.lax.scan(lambda s, k: self.step(s, k), state, jr.split(key, self.dev_iters))
        return state, states
    # ---
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
        genes = jax.vmap(self.genes_shaper)(state.S)
        S_syn, S_sensory, S_motor, S_neurons = genes["synaptic"], genes["sensory"], genes["motor"], genes["neuron_params"]
        synaptic_proteins = S_syn@self.W_syn
        get_weigth = lambda O: synaptic_proteins@O@synaptic_proteins.T
        W = jax.vmap(get_weigth)(self.O_syn).sum(0)
        W = jnp.where(state.mask[None]*state.mask[:,None], W, 0.)
        W = jnp.where(jnp.abs(W)>1e-3, W, 0.)
        sensory = self.sensory_activation_fn(S_sensory)
        motor = self.motor_activation_fn(S_motor)
        v = jnp.zeros((self.max_neurons,))
        mask = state.mask
        X = state.X
        if self.network_type=="ctrnn":
            tau, gain, bias = jax.vmap(self.W_neur)(S_neurons).T
            tau = jnp.clip(tau, 0.01)
            gain = jnp.clip(gain, 0.01)
            return CTRNNState(v=v, W=W, mask=mask, x=X, s=sensory, m=motor, tau=tau, gain=gain, b=bias)
        
        elif self.network_type=="rnn":
            bias = jax.vmap(self.W_neur)(S_neurons)
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
        key_init, key_migr = jr.split(key)
        state = self.init(key_init)
        state, _ = self.do_migration(state, key_migr)
        return self.make_network(state)
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    mdl = RAND(nb_neurons=1, network_type="rnn", key=jr.key(2))
    net = mdl(jr.key(2))
    print(jax.tree.map(lambda x: x.shape, net))