from typing import Dict
import logging

from numpy import ndarray

from msense.opt.drivers.driver import Driver
from msense.utils.array_and_dict_utils import dict_to_array_2d
from msense.utils.array_and_dict_utils import normalize_dict_1d, denormalize_dict_1d
from msense.utils.array_and_dict_utils import array_to_dict_1d, dict_to_array_1d
from msense.utils.array_and_dict_utils import concatenate_variable_bounds
from msense.core.discipline import Discipline


MSENSE_HAS_IPOPT = True
try:
    from cyipopt import Problem as IpoptProblem
except ImportError:
    MSENSE_HAS_IPOPT = False

logger = logging.getLogger(__name__)


class IpoptDriver(Driver):
    """
    This Driver subclass implements the functionality needed to solve an OptProblem using Ipopt.
    """

    class _WrappedDiscipline(object):
        """
        This class defines the functions required by the Ipopt Problem class.
        """

        def __init__(self, discipline: Discipline, callback) -> None:
            self.disc = discipline
            self.callback = callback
            self.iter = 0  # Required since Ipopt doesnt return the number of iterations at exit

        def objective(self, x: ndarray) -> float:
            x = array_to_dict_1d(self.disc.input_vars, x)
            obj_value = self.disc.eval(x)[self.disc.output_vars[0].name]
            return obj_value

        def gradient(self, x: ndarray) -> ndarray:
            x = array_to_dict_1d(self.disc.input_vars, x)
            grad = self.disc.differentiate(x)
            grad = dict_to_array_2d(self.disc.input_vars,
                                    [self.disc.output_vars[0]], grad, flatten=True)
            return grad

        def constraints(self, x: ndarray) -> ndarray:
            x = array_to_dict_1d(self.disc.input_vars, x)
            con_values = self.disc.eval(x)
            con_values = dict_to_array_1d(
                self.disc.output_vars[1:], con_values)
            return con_values

        def jacobian(self, x: ndarray) -> ndarray:
            x = array_to_dict_1d(self.disc.input_vars, x)
            con_jac = self.disc.differentiate(x)
            con_jac = dict_to_array_2d(self.disc.input_vars,
                                       self.disc.output_vars[1:], con_jac)
            return con_jac

        def intermediate(self, *args) -> None:
            self.callback()
            self.iter += 1

    def __init__(self, discipline: Discipline, **kwargs):
        if MSENSE_HAS_IPOPT is False:
            logger.error("Package cyipopt is not available.")
            raise ImportError()

        super().__init__(discipline, **kwargs)
        self.options = {"print_level": 0}

    def _convert_result(self, ipopt_result: Dict[str, any]) -> Dict[str, any]:
        result = {}

        result["inputs"] = array_to_dict_1d(
            self.disc.input_vars, ipopt_result["x"])
        if self.disc.use_norm:
            result["inputs"] = denormalize_dict_1d(
                self.disc.input_vars, result["inputs"])

        result["objective"] = ipopt_result["obj_val"]
        result["jac"] = array_to_dict_1d(
            self.disc.input_vars, ipopt_result["g"])
        result["iter"] = 0
        result["message"] = ipopt_result["status_msg"]
        result["converged"] = ipopt_result["status"]

        return result

    def solve(self, input_values: Dict[str, ndarray], use_norm: bool):
        # Normalize the input values if needed,
        # and covert to 1d numpy array
        if use_norm:
            input_values = normalize_dict_1d(
                self.disc.input_vars, input_values)
        input_values = dict_to_array_1d(
            self.disc.input_vars, input_values)

        # Get the design variable and constraint bounds as arrays
        lb, ub, _ = concatenate_variable_bounds(self.disc.input_vars, use_norm)
        cl, cu, _ = concatenate_variable_bounds(
            self.disc.output_vars[1:], use_norm)

        # Create the Ipopt optimization problem
        wrapped_disc = self._WrappedDiscipline(self.disc, self.callback)
        ipopt_nlp = IpoptProblem(
            n=len(lb), m=len(cl),
            problem_obj=wrapped_disc,
            lb=lb, ub=ub, cl=cl, cu=cu)

        # Add the specified options
        self.options["max_iter"] = self.n_iter_max
        self.options["tol"] = self.tol
        for key, val in self.options.items():
            ipopt_nlp.add_option(key, val)

        # Solve the optimization problem using Ipopt
        _, result = ipopt_nlp.solve(input_values)
        result["iter"] = wrapped_disc.iter

        return self._convert_result(result)
