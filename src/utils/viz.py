import matplotlib.pyplot as plt
import jax.numpy as jnp
from sklearn.decomposition import PCA

def render_network(network, node_colors=None, ax=None, wcmap="coolwarm"):
    
    x = network.x
    W = network.W
    mask = network.mask

    cm = plt.cm.get_cmap(wcmap)
    
    if ax is None:
        fig, ax = plt.subplots(1,1)
        
    if node_colors is None:
        node_colors = jnp.ones(network.mask.shape[0])
    node_colors = node_colors[network.mask.astype(bool)] 
    ax.set_xlim(-1.1,1.1)#type:ignore
    ax.set_ylim(-1.1,1.1)#type:ignore

    wmax = abs(network.W).max()
    W_norm = (network.W / wmax) / 2 + 0.5
    circ = plt.Circle([0,0], 1, edgecolor=(1,1,1,0.7), facecolor=(0,)*4, linewidth=4)
    ax.add_patch(circ)#type:ignore
    for i in range(x.shape[0]):
        if not mask[i]: continue
        for j in range(x.shape[0]):
            if not mask[j]: continue
            xi, yi = x[i]
            xj, yj = x[j]
            w = W_norm[i,j]
            alpha = min(max(float(w)*0.5, 0), 1.)
            ax.plot([xi,xj], [yi,yj], color=cm(float(w)), alpha=alpha)#type:ignore
    ax.scatter(*network.x[mask.astype(bool)].T, c=node_colors, s=100)



def draw_phylogenic_tree(states):
    agents = states.agents
    mask = jnp.ravel(agents.alive)
    T, n_agents = agents.alive.shape
    time = jnp.ravel(jnp.repeat(jnp.arange(T)[:,None], n_agents, axis=1))
    time = time[mask]
    prms = agents.prms.reshape((-1,agents.prms.shape[-1]))[mask]
    pca = PCA(n_components=2)
    prms_projected = pca.fit_transform(prms)
    plt.scatter(*prms_projected.T, c=time, cmap="rainbow")
    plt.show()




