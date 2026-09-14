from typing import Callable, Dict, Type
from enum import Enum

from msense.opt.drivers.driver import Driver, DriverResult
from msense.opt.drivers.scipy_driver import ScipyDriver
from msense.opt.drivers.ipopt_driver import IpoptDriver
from msense.opt.drivers.pymoo_driver import PymooDriver


class DriverType(str, Enum):
    SCIPY_DRIVER = "scipy_driver"
    IPOPT_DRIVER = "ipopt_driver"
    PYMOO_DRIVER = "pymoo_driver"


#: Registry of available drivers. A new backend becomes usable by adding it here.
DRIVERS: Dict[str, Type[Driver]] = {
    DriverType.SCIPY_DRIVER.value: ScipyDriver,
    DriverType.IPOPT_DRIVER.value: IpoptDriver,
    DriverType.PYMOO_DRIVER.value: PymooDriver,
}


def register_driver(name: str, cls: Type[Driver]) -> None:
    """
    Register a driver class under a name, so create_driver can build it.
    """
    DRIVERS[name] = cls


def create_driver(problem, type: str = DriverType.SCIPY_DRIVER,
                  n_iter_max: int = 10, tol: float = 1e-6,
                  callback: Callable = None, **options) -> Driver:
    """
    Create a driver for an optimization problem.

    Args:
        problem (OptProblem): The problem to solve.
        type (str, optional): The driver to use. Defaults to DriverType.SCIPY_DRIVER.
        n_iter_max (int, optional): Maximum major iterations. Defaults to 10.
        tol (float, optional): Convergence tolerance. Defaults to 1e-6.
        callback (Callable, optional): Called at the end of each major iteration.
        **options: Passed to the driver, e.g. method="COBYLA" for a ScipyDriver
            or algorithm="NSGA2", pop_size=40 for a PymooDriver.

    Returns:
        Driver: The driver.
    """
    key = type.value if isinstance(type, DriverType) else str(type)
    if key not in DRIVERS:
        raise ValueError(
            f"Unknown driver '{key}'. Available: {sorted(DRIVERS)}.")

    return DRIVERS[key](problem, n_iter_max=n_iter_max, tol=tol,
                        callback=callback, **options)
