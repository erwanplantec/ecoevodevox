def render(self, state: EnvState, ax:Axes|None=None):

    if ax is None:
        ax = plt.figure().add_subplot()
    else:
        ax=ax
    assert ax is not None

    food = state.food # F, X, Y
    F, H, W = food.shape
    agents = state.agents_states
    food_colors = plt.cm.Set2(jnp.arange(food.shape[0])) #type:ignore

    img = jnp.ones((F,H,W,4)) * food_colors[:,None,None]
    img = jnp.clip(jnp.where(food[...,None], img, 0.).sum(0), 0.0, 1.0) #type:ignore
    img = img.at[:,:,-1].set(jnp.any(food, axis=0))

    img = jnp.where(self.walls[...,None], jnp.array([0.5, 0.5, 0.5, 1.0]), img)

    colormap = lambda e: plt.cm.winter((e / (self.cfg.max_energy*2) + 1) /2) #type:ignore
    for a in range(self.cfg.max_agents):
        if not agents.alive[a] : continue
        body = jax.tree.map(lambda x: x[a], agents.body)
        x,y = body.pos
        h = body.heading
        e = agents.energy[a]
        s = body.size
        body = Rectangle((x-s/2,y-s/2), s, s, angle=(h/(2*jnp.pi))*360, 
                 facecolor=colormap(e), rotation_point="center")
        ax.add_patch(body)
        dy, dx = jnp.sin(h), jnp.cos(h)
        ax.arrow(x, y, dx*s/2, dy*s/2)

    ax.imshow(img.transpose(1,0,2), origin="lower")

# ---

def render_states(self, states: list|EnvState, ax: Axes, cam: Camera):

    if isinstance(states, EnvState):
        T = states.time.shape[0]
        states = [jax.tree.map(lambda x:x[t], states) for t in range(T)]

    for state in states:
        self.render(state, ax)
        cam.snap()

    return cam