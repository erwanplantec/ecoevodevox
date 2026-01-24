from .braitenberg import BraitenbergMotorInterface
#from .ciliated import CiliatedMotorInterface
from .core import MotorInterface

motor_interfaces = {
	"braitenberg": BraitenbergMotorInterface,
	#"cilia": CiliatedMotorInterface
}