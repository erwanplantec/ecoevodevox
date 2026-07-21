from .braitenberg import BraitenbergMotorInterface
from .ciliated import CiliatedMotorInterface
from .ciliated_torque import CiliatedTorqueMotorInterface
from .core import MotorInterface

motor_interfaces = {
	"braitenberg": BraitenbergMotorInterface,
	"ciliated": CiliatedMotorInterface,
	"ciliated_torque": CiliatedTorqueMotorInterface,
}
