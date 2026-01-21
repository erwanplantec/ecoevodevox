import jax
import jax.numpy as jnp
from jaxtyping import Float, Float16, Bool
from flax.struct import PyTreeNode

class ChemicalType(PyTreeNode):
    """chemical type definition"""
    diffusion_rate: Float16 #diffusion rate of chemical in environment
    is_sparse: Bool # whether the chemical is sparse (i.e. only present in a few cells)
    emission_rate: Float16 # base probability of emission (used if sparse)

def make_chemical_diffusion_convolution(env_size: tuple[int,int],
                                        diffusion_rates: jax.Array,
                                        flow: jax.Array|None=None):
    """Creates convolution function for chemical diffusion
    
    Args:
        env_size: tuple[int,int]
        diffusion_rates: jax.Array
        flow: jax.Array|None
    
    Returns:
        Callable[[jax.Array], jax.Array]
    """
    # ---
    H, W = env_size
    # ---
    assert (not H%2) and (not W%2), f"world dimsensions must be even, got {H}x{W}"
    # ---
    mH, mW = H//2, W//2
    L = jnp.mgrid[-mH:mH,-mW:mW]
    D = jnp.sum(jnp.square(L), axis=0, keepdims=True) #1,H,W

    if flow is None:
        diffusion_kernels = jnp.exp(-D/diffusion_rates[:,None,None])
    else:
        flow_norm = jnp.linalg.norm(flow)
        unit_flow = flow/flow_norm
        cosines = jnp.sum(L*unit_flow[:,None,None], axis=0) / (jnp.linalg.norm(L,axis=0))
        cosines = (cosines+1.0) * 0.5
        diffusion_kernels = jnp.exp(-D / ( cosines * diffusion_rates[:,None,None] * flow_norm + (1-cosines)*0.1))

    diffusion_kernels = jnp.where(jnp.isnan(diffusion_kernels), 1.0, diffusion_kernels) #FIX THIS 
    
    diffusion_kernels_fft = jnp.fft.fft2(jnp.fft.fftshift(diffusion_kernels, axes=(1,2)))

    @jax.jit
    def _conv(C: Float[jax.Array, "C H W"])->Float[jax.Array, "C H W"]:
        C_fft = jnp.fft.fft2(C)
        res = jnp.real(jnp.fft.ifft2(C_fft * diffusion_kernels_fft))
        return res

    return _conv