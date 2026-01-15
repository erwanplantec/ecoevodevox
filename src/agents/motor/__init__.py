from .braitenberg import BraitenbergSpatiallyEmbeddedMotorInterface, BraitenbergDirectMotorInterface
from .ciliated import CiliatedMotorInterface
from .base import MotorInterface

motor_interfaces = {
	"braitenberg_se": BraitenbergSpatiallyEmbeddedMotorInterface,
	"braitenberg_direct": BraitenbergDirectMotorInterface,
	"cilia": CiliatedMotorInterface
}