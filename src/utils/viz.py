import matplotlib.pyplot as plt

def render_network(network, ax=None, node_colors=None):
    
    x = network.x
    W = network.W
    mask = network.mask
    id_ = network.id_
    
    if ax is None:
        fig, ax = plt.subplots(1,1)
        
    node_colors = plt.cm.Set1(id_)[network.mask.astype(bool)] if node_colors is None else node_colors[network.mask.astype(bool)]
    ax.scatter(*network.x[mask.astype(bool)].T, c=node_colors, s=100)
    ax.set_xlim(-1.1,1.1)
    ax.set_ylim(-1.1,1.1)

    wmax = abs(network.W).max()
    W_norm = (network.W / wmax) / 2 + 0.5
    circ = plt.Circle([0,0], 1, edgecolor=(1,1,1,0.7), facecolor=(0,)*4, linewidth=4)
    ax.add_patch(circ)
    for i in range(x.shape[0]):
        if not mask[i]: continue
        for j in range(x.shape[0]):
            if not mask[j]: continue
            xi, yi = x[i]
            xj, yj = x[j]
            w = W_norm[i,j]
            alpha = min(max(float(w)*0.5, 0), 1.)
            ax.plot([xi,xj], [yi,yj], color=plt.cm.coolwarm(float(w)), alpha=alpha)