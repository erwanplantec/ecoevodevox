import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import jax.numpy as jnp
from scipy.spatial import Voronoi
from sklearn.decomposition import PCA


def _finite_voronoi_polygons_2d(centroids, radius=None):
    """Voronoi tessellation of `centroids` with the unbounded outer regions closed off.

    `scipy`'s Voronoi leaves the regions on the convex hull open (their vertex list
    contains -1). Each such region is closed by projecting its open ridges out to
    `radius`, far enough that the axis limits clip them back to the archive bounds.

    Returns:
        (regions, vertices) where each region is a list of indices into `vertices`.
    """
    vor = Voronoi(centroids)
    if radius is None:
        radius = np.ptp(vor.points, axis=0).max() * 2

    center = vor.points.mean(axis=0)
    vertices = vor.vertices.tolist()

    # ridges incident to each input point
    ridges: dict[int, list[tuple[int, int, int]]] = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        ridges.setdefault(p1, []).append((p2, v1, v2))
        ridges.setdefault(p2, []).append((p1, v1, v2))

    regions = []
    for p1, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        if all(v >= 0 for v in region):  # already bounded
            regions.append(region)
            continue

        new_region = [v for v in region if v >= 0]
        for p2, v1, v2 in ridges[p1]:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:  # this ridge is finite
                continue
            # project the open ridge outwards, away from the centre
            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            new_region.append(len(vertices))
            vertices.append((vor.vertices[v2] + direction * radius).tolist())

        # fill() needs the polygon's vertices in angular order
        vs = np.asarray([vertices[v] for v in new_region])
        angles = np.arctan2(vs[:, 1] - vs[:, 1].mean(), vs[:, 0] - vs[:, 0].mean())
        regions.append([new_region[i] for i in np.argsort(angles)])

    return regions, np.asarray(vertices)


def plot_repertoire(repertoire, ax=None, cmap="viridis", vmin=None, vmax=None,
                    descriptor_names=None, colorbar=True, title=None,
                    show_elites=False, empty_color=(0.94, 0.94, 0.94),
                    edgecolor="white", linewidth=0.4):
    """Plot a 2D MAP-Elites archive as a Voronoi tessellation coloured by fitness.

    Pure matplotlib, so it does not drag in qdax's plotting stack. Works for any
    centroid layout (regular `compute_euclidean_centroids` grids and CVT centroids
    alike), since the cells are derived from the centroids themselves.

    Args:
        repertoire: anything exposing `centroids` [C, 2], `fitnesses` [C] and, if
            `show_elites`, `descriptors` [C, 2]. Unoccupied cells are those whose
            fitness is not finite (-inf).
        ax: axis to draw on, created if None.
        cmap: colormap over fitness.
        vmin, vmax: fitness colour range; taken from the occupied cells if None.
        descriptor_names: pair of axis labels.
        colorbar: draw a fitness colorbar.
        title: optional axis title.
        show_elites: also scatter each elite's actual descriptor inside its cell.
        empty_color: fill for cells with no elite.
        edgecolor, linewidth: cell borders.

    Returns:
        The axis drawn on.
    """
    centroids = np.asarray(repertoire.centroids, dtype=np.float64)
    fitnesses = np.asarray(repertoire.fitnesses, dtype=np.float64)

    assert centroids.ndim == 2 and centroids.shape[1] == 2, \
        f"plot_repertoire handles 2D descriptor spaces only, got centroids {centroids.shape}"
    assert fitnesses.shape[0] == centroids.shape[0], \
        f"got {fitnesses.shape[0]} fitnesses for {centroids.shape[0]} centroids"

    occupied = np.isfinite(fitnesses)
    if vmin is None:
        vmin = float(fitnesses[occupied].min()) if occupied.any() else 0.0
    if vmax is None:
        vmax = float(fitnesses[occupied].max()) if occupied.any() else 1.0
    # a flat archive (every elite equal, e.g. the `constant` fitness) has no range
    if vmax <= vmin:
        vmax = vmin + 1e-8

    norm = Normalize(vmin=vmin, vmax=vmax)
    colormap = plt.get_cmap(cmap)

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5, 4))

    regions, vertices = _finite_voronoi_polygons_2d(centroids)
    for region, fitness, is_occ in zip(regions, fitnesses, occupied):
        polygon = vertices[region]
        color = colormap(norm(fitness)) if is_occ else empty_color
        ax.fill(*zip(*polygon), facecolor=color, edgecolor=edgecolor, linewidth=linewidth)

    if show_elites and occupied.any():
        descriptors = np.asarray(repertoire.descriptors)[occupied]
        ax.scatter(descriptors[:, 0], descriptors[:, 1], c="k", s=4, alpha=0.5, linewidths=0)

    # clip the outer cells back to the archive bounds
    ax.set_xlim(centroids[:, 0].min(), centroids[:, 0].max())
    ax.set_ylim(centroids[:, 1].min(), centroids[:, 1].max())
    ax.set_aspect("equal", adjustable="box")

    if descriptor_names is not None:
        ax.set_xlabel(descriptor_names[0])
        ax.set_ylabel(descriptor_names[1])
    if title is not None:
        ax.set_title(title)
    if colorbar:
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
        ax.figure.colorbar(mappable, ax=ax, label="fitness")  #type:ignore

    return ax


def render_developmental_trajectory(trajectory, ax=None, cmap="viridis", every=1,
                                    s=6, alpha=0.6, colorbar=True, title=None):
    """Scatter neuron positions over development, coloured by developmental step.

    Args:
        trajectory: a network/developmental state whose leaves carry a leading time
            dimension, i.e. `x` is [T, N, 2] and `mask` is [T, N]. This is what a
            `lax.scan` over development stacks up, e.g. `RAND_CTRNN.do_migration`'s
            trace (`trace["state"]`).
        ax: axis to draw on, created if None.
        cmap: colormap over the step index.
        every: plot only every k-th step (dev_iters is often in the hundreds).
        s, alpha: scatter point size / opacity.
        colorbar: draw a colorbar for the step index.
        title: optional axis title.

    Returns:
        The axis drawn on.
    """
    x = np.asarray(trajectory.x)
    mask = np.asarray(trajectory.mask).astype(bool)

    assert x.ndim == 3 and x.shape[-1] == 2, \
        f"expected x of shape [T, N, 2] (leading time dim), got {x.shape}"
    assert mask.shape == x.shape[:2], \
        f"expected mask of shape {x.shape[:2]}, got {mask.shape}"

    T = x.shape[0]
    steps = np.repeat(np.arange(T)[:, None], x.shape[1], axis=1)  # [T, N]

    # dead/unborn neurons are parked at the origin by the developmental step, so
    # they have to be dropped rather than drawn
    x, mask, steps = x[::every], mask[::every], steps[::every]
    points, point_steps = x[mask], steps[mask]

    if ax is None:
        _, ax = plt.subplots(1, 1)

    ax.set_xlim(-1.1, 1.1)  #type:ignore
    ax.set_ylim(-1.1, 1.1)  #type:ignore
    ax.set_aspect("equal")  #type:ignore
    circ = plt.Circle([0, 0], 1, edgecolor=(1, 1, 1, 0.7), facecolor=(0,) * 4, linewidth=4)
    ax.add_patch(circ)  #type:ignore

    sc = ax.scatter(points[:, 0], points[:, 1], c=point_steps, cmap=cmap,  #type:ignore
                    s=s, alpha=alpha, linewidths=0, vmin=0, vmax=max(T - 1, 1))

    if title is not None:
        ax.set_title(title)  #type:ignore
    if colorbar and points.shape[0]:
        ax.figure.colorbar(sc, ax=ax, label="developmental step")  #type:ignore

    return ax


def render_network(network, node_colors=None, ax=None, wcmap="coolwarm"):
    
    x = network.x
    W = network.W
    mask = network.mask

    cm = plt.get_cmap(wcmap)   # plt.cm.get_cmap was removed in matplotlib 3.9
    
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






def _body_frame_to_world(x_norm, pos, heading, size):
    """Map normalised neuron coords [-1,1]^2 into world coords.

    Mirrors `AgentInterface.get_body_points` exactly: it rotates body-frame offsets by
    `heading - pi/2` and scales them by the body size, with its sample grid spanning [-0.5, 0.5].
    Neuron positions span [-1, 1], hence the extra factor of 1/2 — get this wrong and the network
    drifts out of the body it is supposed to sit in.
    """
    a = heading - np.pi / 2
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return pos[None, :] + (R @ (np.asarray(x_norm).T * size / 2.0)).T


def animate_mini_rollout(states, channel=0, stride=1, fps=20, zoom=1.8,
                         trail=True, node_cmap="coolwarm", weight_cmap="coolwarm",
                         figsize=(11.5, 5.5), path=None):
    """Animate a `MiniEnv`/`MiniTaxis` rollout: the agent moving and turning, network included.

    Two panels, because one cannot show both: an agent is a couple of cells across in a 64-cell
    arena, so its neurons would be sub-pixel in a whole-arena view.
      * left  — the whole arena: sensory field, beacon, the path travelled so far, and the agent
                drawn as an oriented body with a heading tick.
      * right — a zoomed view that follows the agent, showing the body rotating with its grown
                network inside. Neurons are coloured by live activation `v`, so internal dynamics
                are visible, and synapses by weight sign.

    Args:
        states: stacked rollout states, i.e. what `MiniEnv.rollout` returns (leading time axis).
        channel: which field channel to draw as the background (matches `MiniTaxis.channel`).
        stride: keep every `stride`-th frame — the cheapest way to shorten a long rollout.
        fps: playback rate, and the rate used when saving.
        zoom: half-width of the follow view, in body sizes.
        trail: draw the path travelled so far on the left panel.
        path: optional file to save to. `.gif` uses Pillow (no external binary); other suffixes
            (e.g. `.mp4`) need ffmpeg installed.

    Returns:
        `matplotlib.animation.FuncAnimation`. In a notebook, display it with
        `HTML(anim.to_jshtml())`, or pass `path=` to write a file.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Polygon

    agent = states.agent_state
    net = agent.neural_state
    if not hasattr(net, "x") or getattr(net, "mask", None) is None:
        raise ValueError("animate_mini_rollout needs a spatially-embedded network (neural_state.x / .mask)")

    sl = slice(None, None, stride)
    pos = np.asarray(agent.body.pos, np.float32)[sl]          # [T, 2]
    heading = np.asarray(agent.body.heading, np.float32)[sl]  # [T]
    size = np.asarray(agent.body.size, np.float32)[sl]        # [T]
    xs = np.asarray(net.x, np.float32)[sl]                    # [T, N, 2]
    mask = np.asarray(net.mask)[sl].astype(bool)              # [T, N]
    W = np.asarray(net.W, np.float32)[sl]                     # [T, N, N]
    v = np.asarray(net.v, np.float32)[sl] if hasattr(net, "v") else None
    field = np.asarray(states.state_grid, np.float32)[0, channel]   # static over the rollout
    H, Wd = field.shape
    T = pos.shape[0]

    fig, (ax_w, ax_z) = plt.subplots(1, 2, figsize=figsize)
    for ax in (ax_w, ax_z):
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])

    # --- left: whole arena -------------------------------------------------
    # field axis 1 indexes pos[0] and axis 2 indexes pos[1] (see MiniEnv.get_observation), so
    # transpose to put x on the horizontal axis
    ax_w.imshow(field.T, origin="lower", extent=(0, H, 0, Wd), cmap="viridis", alpha=0.85)
    ax_w.set_xlim(0, H); ax_w.set_ylim(0, Wd)
    if hasattr(states, "source"):
        src = np.asarray(states.source, np.float32)[0]
        ax_w.scatter([src[0]], [src[1]], marker="*", s=260, c="#ffd54f",
                     edgecolors="black", linewidths=0.6, zorder=5, label="beacon")
    ax_w.plot(pos[:, 0], pos[:, 1], color="white", alpha=0.18, lw=1)   # full path, faint
    (trail_line,) = ax_w.plot([], [], color="#ff5252", lw=1.6, alpha=0.9)
    body_w = Polygon(np.zeros((4, 2)), closed=True, facecolor="#ff5252", alpha=0.55,
                     edgecolor="white", lw=1.2, zorder=6)
    ax_w.add_patch(body_w)
    (head_w,) = ax_w.plot([], [], color="white", lw=1.8, zorder=7)
    ax_w.set_title("arena", fontsize=10)

    # --- right: agent-centred zoom with the network ------------------------
    body_z = Polygon(np.zeros((4, 2)), closed=True, facecolor=(1, 1, 1, 0.06),
                     edgecolor="#90a4ae", lw=2.0, zorder=1)
    ax_z.add_patch(body_z)
    (head_z,) = ax_z.plot([], [], color="#90a4ae", lw=2.5, zorder=2)
    edges = LineCollection([], zorder=3)
    ax_z.add_collection(edges)
    # seed with an empty array rather than no `c`, or matplotlib drops the colormap
    nodes = ax_z.scatter(np.zeros(0), np.zeros(0), c=np.zeros(0), s=90, zorder=4,
                         edgecolors="black", linewidths=0.5, cmap=node_cmap)
    ax_z.set_facecolor("#101018")
    ax_z.set_title("body-fixed network (colour = activation v)", fontsize=10)

    # synapse colours are constant: the grown network is fixed for the agent's lifetime
    live0 = np.flatnonzero(mask[0])
    wmax = float(np.abs(W[0][np.ix_(live0, live0)]).max()) if live0.size else 1.0
    wmax = wmax or 1.0
    vmax = float(np.abs(v).max()) if v is not None and v.size else 1.0
    vmax = vmax or 1.0
    nodes.set_clim(-vmax, vmax)
    cmw = plt.get_cmap(weight_cmap)

    # square body outline in normalised body coords
    corners = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])

    def frame(t):
        p, h, s = pos[t], heading[t], size[t]
        quad = _body_frame_to_world(corners, p, h, s)
        body_w.set_xy(quad); body_z.set_xy(quad)
        # heading tick: from body centre toward the front (+y in body frame)
        tip = _body_frame_to_world(np.array([[0., 0.], [0., 1.]]), p, h, s)
        head_w.set_data(tip[:, 0], tip[:, 1]); head_z.set_data(tip[:, 0], tip[:, 1])
        if trail:
            trail_line.set_data(pos[:t + 1, 0], pos[:t + 1, 1])

        live = np.flatnonzero(mask[t])
        pts = _body_frame_to_world(xs[t][live], p, h, s) if live.size else np.zeros((0, 2))
        if live.size:
            sub = W[t][np.ix_(live, live)]
            ii, jj = np.nonzero(np.abs(sub) >= 1e-3)
            segs = np.stack([pts[jj], pts[ii]], axis=1)          # j -> i
            ww = sub[ii, jj]
            edges.set_segments(segs)
            edges.set_color(cmw(0.5 + 0.5 * np.clip(ww / wmax, -1, 1)))
            edges.set_alpha(0.5)
            edges.set_linewidth(1.0)
            nodes.set_offsets(pts)
            nodes.set_array(v[t][live] if v is not None else np.zeros(live.size))
        else:
            edges.set_segments([]); nodes.set_offsets(np.zeros((0, 2)))

        span = zoom * max(float(s), 1e-3)
        ax_z.set_xlim(p[0] - span, p[0] + span); ax_z.set_ylim(p[1] - span, p[1] + span)
        fig.suptitle(f"step {t * stride}   pos ({p[0]:.1f}, {p[1]:.1f})   "
                     f"heading {np.degrees(h) % 360:.0f}°   neurons {live.size}", fontsize=10)
        return body_w, body_z, head_w, head_z, edges, nodes, trail_line

    anim = FuncAnimation(fig, frame, frames=T, interval=1000 / fps, blit=False)
    if path is not None:
        if str(path).endswith(".gif"):
            anim.save(path, writer=PillowWriter(fps=fps))
        else:
            anim.save(path, fps=fps)
        plt.close(fig)
    return anim


def animate_multitaxis_rollout(states, dev_trajectory=None, channel=0, stride=1, fps=20,
                               node_cmap="coolwarm", weight_cmap="coolwarm",
                               figsize=(11.5, 5.5), path=None):
    """Animate a `MiniMultiTaxis` rollout, where the beacon moves whenever the agent reaches it.

    Written for the sequential task specifically: `animate_mini_rollout` reads the field and
    beacon once (they are static in `MiniTaxis`), so it would show only the first beacon here.
    This reads `state_grid`, `source` and `n_reached` per frame, so the field re-centres and a
    fresh star appears each time a beacon is consumed.

      * left  — the arena: the *current* beacon's field and star, the path so far (recoloured at
                each beacon so successive legs are distinguishable), the agent as an oriented
                body, and a "reached N" counter.
      * right — the network in the **fixed body frame**, heading up and not rotating: neurons sit
                at their grown positions and only their colour changes with activation `v`. Use
                this to read the internal dynamics without the body's motion distracting; use the
                left panel for where the agent actually points.

    If `dev_trajectory` is given, the animation plays in two acts: first the network **grows**
    (development runs on the right panel while the agent waits at its spawn on the left), then the
    rollout proceeds as above. Omit it and the animation is exactly the rollout.

    Args:
        states: stacked `MiniMultiTaxis.rollout` output (leading time axis; needs `source`,
            `n_reached`, and a per-step `state_grid`).
        dev_trajectory: optional developmental trace — a network state whose leaves carry a
            leading dev-step axis (`x` is [D, N, 2], `mask` [D, N]; `W`/`v` used if present), i.e.
            the same object `render_developmental_trajectory` takes. Shown before the rollout.
        channel: field channel to draw (matches `MiniMultiTaxis.channel`).
        stride: keep every `stride`-th frame (applied to development and rollout alike).
        path: optional output file. `.gif` uses Pillow; other suffixes need ffmpeg.

    Returns:
        `matplotlib.animation.FuncAnimation` (display with `HTML(anim.to_jshtml())` or save via
        `path=`).
    """
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Polygon

    agent = states.agent_state
    net = agent.neural_state
    if not hasattr(net, "x") or getattr(net, "mask", None) is None:
        raise ValueError("animate_multitaxis_rollout needs a spatially-embedded network")
    # `n_reached` is what MiniMultiTaxis adds over MiniTaxis; a plain MiniTaxis rollout (single,
    # fixed beacon) lacks it and should go through animate_mini_rollout instead
    if not hasattr(states, "n_reached"):
        raise ValueError("states has no `n_reached`; this is not a MiniMultiTaxis rollout — "
                         "use animate_mini_rollout for the single-beacon MiniTaxis")

    sl = slice(None, None, stride)
    pos = np.asarray(agent.body.pos, np.float32)[sl]          # [T, 2]
    heading = np.asarray(agent.body.heading, np.float32)[sl]  # [T]
    size = np.asarray(agent.body.size, np.float32)[sl]        # [T]
    xs = np.asarray(net.x, np.float32)[sl]                    # [T, N, 2]
    mask = np.asarray(net.mask)[sl].astype(bool)              # [T, N]
    W = np.asarray(net.W, np.float32)[sl]                     # [T, N, N]
    v = np.asarray(net.v, np.float32)[sl] if hasattr(net, "v") else None
    fields = np.asarray(states.state_grid, np.float32)[sl, channel]   # [T, H, W]
    source = np.asarray(states.source, np.float32)[sl]        # [T, 2]
    n_reached = np.asarray(states.n_reached)[sl]              # [T]
    H, Wd = fields.shape[1:]
    T = pos.shape[0]

    # optional development phase, played before the rollout on the same right panel
    if dev_trajectory is not None:
        if not hasattr(dev_trajectory, "x") or getattr(dev_trajectory, "mask", None) is None:
            raise ValueError("dev_trajectory needs `x` [D, N, 2] and `mask` [D, N] with a leading "
                             "dev-step axis (as render_developmental_trajectory expects)")
        d_x = np.asarray(dev_trajectory.x, np.float32)[sl]                 # [D, N, 2]
        d_mask = np.asarray(dev_trajectory.mask)[sl].astype(bool)         # [D, N]
        d_W = np.asarray(dev_trajectory.W, np.float32)[sl] if hasattr(dev_trajectory, "W") else None
        d_v = np.asarray(dev_trajectory.v, np.float32)[sl] if hasattr(dev_trajectory, "v") else None
        D = d_x.shape[0]
    else:
        d_x = d_mask = d_W = d_v = None
        D = 0

    # a new beacon leg starts wherever the source jumps; used to recolour the path
    leg = np.concatenate([[0], np.cumsum(np.any(np.diff(source, axis=0) != 0, axis=-1))])
    leg_colors = plt.get_cmap("tab10")

    fig, (ax_w, ax_z) = plt.subplots(1, 2, figsize=figsize)
    for ax in (ax_w, ax_z):
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

    # --- left: arena, field + beacon both dynamic ---------------------------
    im = ax_w.imshow(fields[0].T, origin="lower", extent=(0, H, 0, Wd), cmap="viridis",
                     alpha=0.85, vmin=float(fields.min()), vmax=float(fields.max()))
    ax_w.set_xlim(0, H); ax_w.set_ylim(0, Wd)
    beacon = ax_w.scatter([source[0, 0]], [source[0, 1]], marker="*", s=280, c="#ffd54f",
                          edgecolors="black", linewidths=0.7, zorder=6)
    # one line per leg, filled in as the run progresses
    trail_lines = [ax_w.plot([], [], color=leg_colors(L % 10), lw=1.8, alpha=0.9)[0]
                   for L in range(int(leg.max()) + 1)]
    body_w = Polygon(np.zeros((4, 2)), closed=True, facecolor="#ff5252", alpha=0.6,
                     edgecolor="white", lw=1.2, zorder=7)
    ax_w.add_patch(body_w)
    (head_w,) = ax_w.plot([], [], color="white", lw=1.8, zorder=8)
    ax_w.set_title("arena", fontsize=10)

    # --- right: network in the fixed body frame (heading up, no rotation) ----
    ax_z.set_xlim(-1.15, 1.15); ax_z.set_ylim(-1.15, 1.15)
    ax_z.set_facecolor("#101018")
    ax_z.add_patch(Polygon(np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float), closed=True,
                           facecolor=(1, 1, 1, 0.05), edgecolor="#90a4ae", lw=2.0, zorder=1))
    ax_z.annotate("front", (0, 1.02), ha="center", va="bottom", color="#90a4ae", fontsize=8)
    edges = LineCollection([], zorder=3); ax_z.add_collection(edges)
    nodes = ax_z.scatter(np.zeros(0), np.zeros(0), c=np.zeros(0), s=110, zorder=4,
                         edgecolors="black", linewidths=0.5, cmap=node_cmap)
    ax_z.set_title("body-fixed network — heading up (colour = activation v)", fontsize=10)

    # colour scales span both phases so a neuron's colour means the same thing throughout. Weights
    # are read over live neurons only (dead RAND cells can carry arbitrary W entries).
    def _wmax(Wm, mk):
        lv = np.flatnonzero(mk)
        return float(np.abs(Wm[np.ix_(lv, lv)]).max()) if lv.size and Wm is not None else 0.0
    wmax = _wmax(W[0], mask[0])
    vmax = float(np.abs(v).max()) if v is not None and v.size else 0.0
    if dev_trajectory is not None:
        if d_W is not None:
            wmax = max(wmax, _wmax(d_W[-1], d_mask[-1]))
        if d_v is not None and d_v.size:
            vmax = max(vmax, float(np.abs(d_v).max()))
    wmax = wmax or 1.0
    vmax = vmax or 1.0
    nodes.set_clim(-vmax, vmax)
    cmw = plt.get_cmap(weight_cmap)
    corners = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])

    def draw_network(x_frame, mask_frame, W_frame, v_frame) -> int:
        """Right panel: neurons at their body-frame positions, edges by weight sign. Shared by the
        development and rollout phases (only the data source differs). Returns the live count."""
        live = np.flatnonzero(mask_frame)
        bpts = x_frame[live] if live.size else np.zeros((0, 2))
        if live.size and W_frame is not None:
            sub = W_frame[np.ix_(live, live)]
            ii, jj = np.nonzero(np.abs(sub) >= 1e-3)
            edges.set_segments(np.stack([bpts[jj], bpts[ii]], axis=1))    # j -> i
            edges.set_color(cmw(0.5 + 0.5 * np.clip(sub[ii, jj] / wmax, -1, 1)))
            edges.set_alpha(0.5); edges.set_linewidth(1.0)
        else:
            edges.set_segments([])
        if live.size:
            nodes.set_offsets(bpts)
            nodes.set_array(v_frame[live] if v_frame is not None else np.zeros(live.size))
        else:
            nodes.set_offsets(np.zeros((0, 2)))
        return int(live.size)

    def _draw_body(p, h, s):
        quad = _body_frame_to_world(corners, p, h, s); body_w.set_xy(quad)
        tip = _body_frame_to_world(np.array([[0., 0.], [0., 1.]]), p, h, s)
        head_w.set_data(tip[:, 0], tip[:, 1])

    def frame(k):
        if k < D:
            # development: the agent waits at its spawn while its network grows
            im.set_data(fields[0].T); beacon.set_offsets(source[0])
            _draw_body(pos[0], heading[0], size[0])
            for line in trail_lines:
                line.set_data([], [])
            n = draw_network(d_x[k], d_mask[k],
                             d_W[k] if d_W is not None else None,
                             d_v[k] if d_v is not None else None)
            ax_z.set_title("development — growing (colour = activation v)", fontsize=10)
            fig.suptitle(f"development  step {k * stride} / {(D - 1) * stride}   neurons {n}",
                         fontsize=10)
        else:
            t = k - D
            p, h, s = pos[t], heading[t], size[t]
            im.set_data(fields[t].T); beacon.set_offsets(source[t])
            _draw_body(p, h, s)
            for L, line in enumerate(trail_lines):
                m = (leg[:t + 1] == L)
                line.set_data(pos[:t + 1][m, 0], pos[:t + 1][m, 1])
            n = draw_network(xs[t], mask[t], W[t], v[t] if v is not None else None)
            ax_z.set_title("body-fixed network — heading up (colour = activation v)", fontsize=10)
            fig.suptitle(f"step {t * stride}   reached {int(n_reached[t])}   "
                         f"pos ({p[0]:.1f}, {p[1]:.1f})   heading {np.degrees(h) % 360:.0f}°   "
                         f"neurons {n}", fontsize=10)
        return (im, beacon, body_w, head_w, edges, nodes, *trail_lines)

    anim = FuncAnimation(fig, frame, frames=D + T, interval=1000 / fps, blit=False)
    if path is not None:
        if str(path).endswith(".gif"):
            anim.save(path, writer=PillowWriter(fps=fps))
        else:
            anim.save(path, fps=fps)
        plt.close(fig)
    return anim
