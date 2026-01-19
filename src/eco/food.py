import jax
import jax.numpy as jnp
from jaxtyping import Float, Float16, Float32, Bool
from flax.struct import PyTreeNode

type FoodMap = Bool[jax.Array, "F H W"]

class FoodType(PyTreeNode):
    """food type definition"""
    growth_rate: Float32 # growth rate of food
    dmin: Float32 # minimum distance from food source to start growing
    dmax: Float32 # maximum distance from food source to stop growing
    chemical_signature: Float32 # chemical signature of food
    energy_concentration: Float32 # energy concentration of food
    spontaneous_grow_prob: Float32 # probability of spontaneous growth
    initial_density: Float32 # initial density of food

def make_growth_convolution(env_size: tuple[int,int],
                            reproduction_rates: jax.Array,
                            dmins: jax.Array,
                            dmaxs: jax.Array,
                            inhib: float=-1.0,
                            dtype: type=jnp.float32):
    """Creates convolution function for food growth probabilities using fft convolution"""
    # ---
    H, W = env_size
    # ---
    assert (not H%2) and (not W%2), f"world dimsensions must be even, got {H}x{W}"
    # ---
    mH, mW = H//2, W//2
    L = jnp.mgrid[-mH:mH,-mW:mW]
    D = jnp.linalg.norm(L, axis=0, keepdims=True)

    growth_kernels = ((D>=dmins[:,None,None]) & (D<=dmaxs[:,None,None])).astype(jnp.float32)
    growth_kernels = growth_kernels / growth_kernels.sum(axis=(1,2), keepdims=True)
    growth_kernels = growth_kernels * reproduction_rates[:,None,None]
    growth_kernels = jnp.where(D<dmins[:,None,None], inhib, growth_kernels); assert isinstance(growth_kernels,jax.Array)
    growth_kernels_fft = jnp.fft.fft2(jnp.fft.fftshift(growth_kernels, axes=(1,2))).astype(dtype)

    @jax.jit
    def _conv(F: Bool[jax.Array, "F H W"])->jax.Array:
        F_fft = jnp.fft.fft2(F.astype(dtype))
        P = jnp.real(jnp.fft.ifft2(F_fft*growth_kernels_fft))
        P = jnp.where((P<0)|jnp.isclose(P,0.0), 0.0, P); assert isinstance(P, jax.Array)
        return P

    return _conv