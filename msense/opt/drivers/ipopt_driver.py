from typing import Dict

from numpy import ndarray
from cyipopt import Problem as IpoptProblem

from msense.core.discipline import Discipline
from msense.utils.array_and_dict_utils import concatenate_variable_bounds
from msense.utils.array_and_dict_utils import array_to_dict_1d, dict_to_array_1d
from msense.utils.array_and_dict_utils import normalize_dict_1d
from msense.utils.array_and_dict_utils import dict_to_array_2d
from msense.opt.drivers.driver import Driver


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

        def objective(self, x: ndarray) -> float:
            x = array_to_dict_1d(self.disc.input_vars, x)
            value = self.disc.eval(x)[self.disc.output_vars[0].name]
            return value

        def gradient(self, x: ndarray) -> ndarray:
            x = array_to_dict_1d(self.disc.input_vars, x)
            grad = self.disc.differentiate(x)
            grad = dict_to_array_2d(self.disc.input_vars,
                                    [self.disc.output_vars[0]], grad, flatten=True)
            return grad

        def constraints(self, x: ndarray) -> ndarray:
            x = array_to_dict_1d(self.disc.input_vars, x)
            cons = self.disc.eval(x)
            cons = dict_to_array_1d(self.disc.output_vars[1:], cons)
            return cons

        def jacobian(self, x: ndarray) -> ndarray:
            x = array_to_dict_1d(self.disc.input_vars, x)
            jac = self.disc.differentiate(x)
            jac = dict_to_array_2d(self.disc.input_vars,
                                   self.disc.output_vars[1:], jac)
            return jac

        def intermediate(self, *args) -> None:
            self.callback()

    def __init__(self, discipline: Discipline, **kwargs):
        super().__init__(discipline, **kwargs)
        self.options = {"print_level": 0}

    def _convert_result(self, _result) -> Dict[str, any]:
        pass

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
        ipopt_nlp = IpoptProblem(
            n=len(lb), m=len(cl),
            problem_obj=self._WrappedDiscipline(self.disc, self.callback),
            lb=lb, ub=ub, cl=cl, cu=cu)

        # Add the specified options
        self.options["max_iter"] = self.n_iter_max
        self.options["tol"] = self.tol
        for key, val in self.options.items():
            ipopt_nlp.add_option(key, val)

        # Solve the optimization problem using Ipopt
        result = ipopt_nlp.solve(input_values)

        return self._convert_result(result)
