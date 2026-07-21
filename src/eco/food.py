import warnings

import jax
import jax.numpy as jnp
from jaxtyping import Float, Float16, Float32, Bool, Array
from flax.struct import PyTreeNode

type FoodMap = Bool[jax.Array, "F H W"]


# functions a growth expression may call; deliberately small — a growth spec describes a smooth
# field over the grid, not arbitrary computation
_FIELD_FUNCS = {
    "sin": jnp.sin, "cos": jnp.cos, "tan": jnp.tan, "exp": jnp.exp, "log": jnp.log,
    "sqrt": jnp.sqrt, "abs": jnp.abs, "tanh": jnp.tanh, "sign": jnp.sign,
    "minimum": jnp.minimum, "maximum": jnp.maximum, "clip": jnp.clip, "where": jnp.where,
    "pi": jnp.pi,
}


def eval_growth_field(spec, grid_size: tuple[int, int]) -> jnp.ndarray:
    """Turn a food-growth spec into a `[H, W]` per-cell growth-rate field.

    `spec` is either a number (uniform field) or a string expression of the grid coordinates
    ``x`` and ``y``, e.g. ``"0.005 * (1 - y)"``. Coordinates are **normalised to [0, 1)** so a
    config means the same thing at any resolution, and follow the world's axis order: ``x`` runs
    along the first spatial axis (``Body.pos[0]``), ``y`` along the second (``Body.pos[1]``) — the
    axes an agent moves along.

    The world is toroidal, so a non-periodic expression like ``1 - y`` has a seam at the y wrap
    (it jumps from ~1 back to 0); use a periodic form such as ``0.5*(1 + sin(2*pi*y))`` for a
    smooth gradient. Negative values are clipped to 0 (a negative growth rate is meaningless) with
    a warning, since that usually means a sign error.

    Only ``x``, ``y`` and a small set of math functions are in scope; the expression comes from
    the config, which is author-trusted, but the namespace is restricted so a typo fails loudly.
    """
    H, W = grid_size
    if isinstance(spec, (int, float)):
        return jnp.full((H, W), float(spec), jnp.float32)
    if not isinstance(spec, str):
        raise TypeError(f"growth_rate must be a number or an expression string, got {spec!r}")

    # x varies along axis 0 (H), y along axis 1 (W); [0, 1) matches the toroidal wrap
    x, y = jnp.meshgrid(jnp.arange(H) / H, jnp.arange(W) / W, indexing="ij")
    try:
        field = eval(spec, {"__builtins__": {}}, {**_FIELD_FUNCS, "x": x, "y": y})
    except Exception as e:
        raise ValueError(f"could not evaluate growth_rate expression {spec!r}: {e}") from e
    field = jnp.broadcast_to(jnp.asarray(field, jnp.float32), (H, W))   # constants -> full grid
    if float(field.min()) < 0.0:
        warnings.warn(f"growth_rate {spec!r} is negative somewhere (min {float(field.min()):.3g}); "
                      "clipping to 0 — check the sign of the expression")
        field = jnp.clip(field, 0.0, None)
    return field


class FoodType(PyTreeNode):
    """food type definition"""
    growth_field: Float32 # per-cell growth rate, [F, H, W] (uniform when the spec is a constant)
    dmin: Float32 # minimum distance from food source to start growing
    dmax: Float32 # maximum distance from food source to stop growing
    chemical_signature: Float32 # chemical signature of food
    energy_concentration: Float32 # energy concentration of food
    spontaneous_grow_prob: Float32 # probability of spontaneous growth
    initial_density: Float32 # initial density of food

def make_growth_convolution(env_size: tuple[int,int],
                            growth_field: jax.Array,
                            dmins: jax.Array,
                            dmaxs: jax.Array,
                            inhib: float=-1.0,
                            dtype: type=jnp.float32):
    """Creates convolution function for food growth probabilities using fft convolution.

    The growth **rate** is separable from the convolution: the kernel measures what fraction of a
    target cell's annulus already holds food (a purely geometric quantity), and the per-cell rate
    just scales it. So the kernel is built at rate 1 and `growth_field` ([F, H, W]) is applied
    pointwise at the target cell, which is what lets the rate vary in space at the cost of one
    elementwise multiply and no extra FFT. A uniform field reproduces the old constant-rate
    behaviour exactly (growth rates are <= 1, so the integer inhibition always dominates the
    sub-1 annulus fraction whether the rate is baked into the kernel or applied afterwards)."""
    # ---
    H, W = env_size
    # ---
    assert (not H%2) and (not W%2), f"world dimsensions must be even, got {H}x{W}"
    # ---
    mH, mW = H//2, W//2
    L = jnp.mgrid[-mH:mH,-mW:mW]
    D = jnp.linalg.norm(L, axis=0, keepdims=True)

    # rate-1 kernel: normalised annulus (sums to 1), with the inner disk set to `inhib`
    growth_kernels = ((D>=dmins[:,None,None]) & (D<=dmaxs[:,None,None])).astype(jnp.float32)
    growth_kernels = growth_kernels / growth_kernels.sum(axis=(1,2), keepdims=True)
    growth_kernels = jnp.where(D<dmins[:,None,None], inhib, growth_kernels); assert isinstance(growth_kernels,jax.Array)
    growth_kernels_fft = jnp.fft.fft2(jnp.fft.fftshift(growth_kernels, axes=(1,2))).astype(dtype)

    field = jnp.asarray(growth_field, dtype)   # [F, H, W]

    # Noise floor for the FFT round-trip. The kernel's positive entries are all `1 / annulus_cells`
    # (rate is applied afterwards), so any real output is an integer multiple of it and the
    # smallest possible non-zero one is that value itself. Anything below half of it cannot be a
    # genuine sum of kernel entries and is round-off.
    #
    # This has to be derived from the kernel rather than fixed: the FFT of a 512x512 grid leaves
    # residuals around 1e-7, which the previous `isclose(P, 0)` guard let through (its atol is
    # 1e-8). Those residuals became growth probabilities, so food appeared spontaneously even with
    # spontaneous_grow_prob = 0 — visibly so, since a noise-born cell then seeds real growth.
    _smallest_real = jnp.where(growth_kernels > 0, growth_kernels, jnp.inf).min(axis=(1,2), keepdims=True)
    noise_floor = 0.5 * _smallest_real

    @jax.jit
    def _conv(F: Bool[jax.Array, "F H W"])->jax.Array:
        F_fft = jnp.fft.fft2(F.astype(dtype))
        P = jnp.real(jnp.fft.ifft2(F_fft*growth_kernels_fft))
        # also drops negatives (inhibited cells), which sit below any positive floor
        P = jnp.where(P < noise_floor, 0.0, P); assert isinstance(P, jax.Array)
        # per-cell growth rate at the target; a region with field 0 grows no food (as growth_rate
        # 0 did before), and the noise floor above already ran on the rate-1 annulus fractions
        P = P * field
        return P

    return _conv