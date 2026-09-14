from typing import Mapping
import logging

from numpy import ndarray

from msense.opt.drivers.driver import Driver, DriverResult
from msense.utils.array_and_dict_utils import dict_to_array_1d, dict_to_array_2d

MSENSE_HAS_IPOPT = True
try:
    from cyipopt import Problem as IpoptProblem
except ImportError:
    MSENSE_HAS_IPOPT = False

logger = logging.getLogger(__name__)


class IpoptDriver(Driver):
    """
    This Driver subclass implements the functionality needed to solve an OptProblem
    using Ipopt.
    """

    supports_multi_objective = False
    supports_nonlinear_constraints = True
    supports_equality_constraints = True
    requires_gradients = True

    class _WrappedProblem(object):
        """
        This class defines the functions required by the Ipopt Problem class.
        All conversion between Ipopt's design vector and the problem's physical
        values goes through the driver.
        """

        def __init__(self, driver) -> None:
            self.driver = driver
            self.prob = driver.prob
            self._latest = None

        def objective(self, x: ndarray) -> float:
            values = self.driver.evaluate(x)
            self._latest = {**self.driver.design_dict(x), **values}
            return self.prob.scalar_objective(values)

        def gradient(self, x: ndarray) -> ndarray:
            return self.prob.scalar_objective_jac(self.driver.differentiate(x))

        def constraints(self, x: ndarray) -> ndarray:
            return dict_to_array_1d([con.var for con in self.prob.constraints],
                                    self.driver.evaluate(x))

        def jacobian(self, x: ndarray) -> ndarray:
            return dict_to_array_2d(self.prob.design_vars,
                                    [con.var for con in self.prob.constraints],
                                    self.driver.differentiate(x))

        def intermediate(self, *args) -> None:
            self.driver._callback(values=self._latest)

    def __init__(self, problem, **kwargs):
        if MSENSE_HAS_IPOPT is False:
            logger.error("Package cyipopt is not available.")
            raise ImportError("cyipopt is required by IpoptDriver.")

        super().__init__(problem, **kwargs)
        self.options.setdefault("print_level", 0)

    def solve(self, x0: Mapping[str, ndarray], use_norm: bool) -> DriverResult:
        x0 = self.starting_array(x0)
        self.iter = 0

        # Design variable and constraint bounds. Ipopt takes the two-sided
        # constraint form directly, so no residual conversion is needed.
        xl, xu = self.prob.design_bounds(use_norm)
        cl, cu = self.prob.constraint_bounds()

        wrapped = self._WrappedProblem(self)
        ipopt_nlp = IpoptProblem(n=len(xl), m=len(cl), problem_obj=wrapped,
                                 lb=xl, ub=xu, cl=cl, cu=cu)

        self.options["max_iter"] = self.n_iter_max
        self.options["tol"] = self.tol
        for key, val in self.options.items():
            ipopt_nlp.add_option(key, val)

        x, result = ipopt_nlp.solve(x0)

        return DriverResult(converged=result["status"] == 0,
                            message=str(result["status_msg"]),
                            x=self.design_dict(x),
                            values=self.evaluate(x),
                            n_eval=self.prob.n_eval,
                            n_iter=self.iter)
