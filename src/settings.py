"""Global numerical settings.

Values that need to stay consistent across the platform live here, so a precision choice is
made in one place rather than being repeated at every site that creates an array.
"""

import jax.numpy as jnp

# dtype of agent body positions and headings.
#
# float16 halves its resolution at every power of two: the gap between representable values
# (ULP) is 0.25 around coordinate 300, 0.5 above 512, and 1.0 above 1024. A motor step smaller
# than half that gap rounds straight back to the same position, so agents simply stop moving
# past a threshold that depends on where they stand. In a 1024x1024 world that freezes
# everything beyond the midpoint and makes the world visibly split into quadrants; it also
# inflates measured distances below the threshold, since real steps get rounded up to the ULP.
#
# float32 has ~7 significant digits, so cell-scale worlds are exact and step sizes down to
# ~1e-4 register anywhere. The memory cost is negligible (2 floats per agent).
POSITION_DTYPE = jnp.float32
