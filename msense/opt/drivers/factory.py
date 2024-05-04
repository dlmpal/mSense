from enum import Enum

from msense.core.discipline import Discipline
from msense.opt.drivers.scipy_driver import ScipyDriver


class DriverType(str, Enum):
    SCIPY_DRIVER = "SciPyDriver"


def create_driver(disc: Discipline, type: DriverType = DriverType.SCIPY_DRIVER):

    if type == DriverType.SCIPY_DRIVER:
        return ScipyDriver(disc)
    else:
        raise ValueError(f"DriverType {type} is not available.")
