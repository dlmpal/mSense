from typing import Callable
from enum import Enum

from msense.core.discipline import Discipline
from msense.opt.drivers.driver import Driver
from msense.opt.drivers.scipy_driver import ScipyDriver
from msense.opt.drivers.ipopt_driver import IpoptDriver


class DriverType(str, Enum):
    SCIPY_DRIVER = "scipy_driver"
    IPOPT_DRIVER = "ipopt_driver"


def create_driver(discipline: Discipline, type: DriverType = DriverType.SCIPY_DRIVER,
                  n_iter_max: int = 10, tol: float = 1e-6, callback: Callable = None, **options) -> Driver:
    kwargs = {"n_iter_max": n_iter_max, "tol": tol, "callback": callback}
    kwargs.update(options)

    type = DriverType(type)

    if type == DriverType.SCIPY_DRIVER:
        return ScipyDriver(discipline, **kwargs)
    elif type == DriverType.IPOPT_DRIVER:
        return IpoptDriver(discipline, **kwargs)
