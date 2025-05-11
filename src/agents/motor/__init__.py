from .braitenberg import BraitenbergMotorInterface
from .ciliated import CiliatedMotorInterface
from .base import MotorInterface

motor_interfaces = {
	"braitenberg": BraitenbergMotorInterface,
	"cilia": CiliatedMotorInterface
}