from .core import SensoryInterface
from .spatially_embedded import SpatiallyEmbeddedSensoryInterface
from .flatten import FlattenSensoryInterface

sensory_interfaces = {
	"spatially_embedded": SpatiallyEmbeddedSensoryInterface,
	"flatten": FlattenSensoryInterface,
}