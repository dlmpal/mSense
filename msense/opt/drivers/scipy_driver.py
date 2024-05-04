from typing import Dict, List, Tuple, Callable

from numpy import ndarray, zeros
from scipy.optimize import NonlinearConstraint, Bounds, minimize, OptimizeResult

from msense.core.constants import FLOAT_DTYPE
from msense.core.variable import Variable
from msense.core.discipline import Discipline
from msense.utils.array_and_dict_utils import array_to_dict_1d, dict_to_array_1d
from msense.utils.array_and_dict_utils import normalize_dict_1d, denormalize_dict_1d
from msense.utils.array_and_dict_utils import get_variable_list_size
from msense.opt.drivers.driver import Driver

ScipyFun = Callable[[ndarray], ndarray]


class ScipyDriver(Driver):
    """
    This Driver subclass implements the functionality needed to solve an OptProblem using SciPy optimizers.  
    """

    def __init__(self, problem: Discipline, method: str = "SLSQP", tol: float = 1e-6, **options):
        super().__init__(problem)
        self.method = method
        self.tol = tol
        self.options = options

    def _wrap_bounds(self, use_norm: bool):
        n_inputs = get_variable_list_size(self.disc.input_vars)
        lb_arr = zeros(n_inputs, FLOAT_DTYPE)
        ub_arr = zeros(n_inputs, FLOAT_DTYPE)
        keep_feasible_arr = zeros(n_inputs, bool)

        idx = 0
        for var in self.disc.input_vars:
            lb, ub, keep_feasible = var.get_bounds_as_array(use_norm)
            lb_arr[idx: idx + var.size] = lb
            ub_arr[idx: idx + var.size] = ub
            keep_feasible_arr[idx: idx + var.size] = keep_feasible
            idx += var.size

        return Bounds(lb_arr, ub_arr, keep_feasible_arr)

    def _wrap_func_and_jac(self, var: Variable, scalar_func: bool = False) -> Tuple[ScipyFun, ScipyFun]:
        def func(x: ndarray):
            x = array_to_dict_1d(self.disc.input_vars, x)
            value = self.disc.eval(x)[var.name]

            if scalar_func:
                value = value[0]

            if self.iter == 0 and var.name == self.disc.output_vars[0].name:
                self.callback()
                self.iter += 1

            return value

        def jac(x: ndarray):
            x = array_to_dict_1d(self.disc.input_vars, x)
            jac = self.disc.differentiate(x)[var.name]
            jac = dict_to_array_1d(self.disc.input_vars, jac)
            return jac

        return func, jac

    def _wrap_constraints(self, use_norm: bool) -> List[NonlinearConstraint]:
        cons = []
        for con in self.disc.output_vars[1:]:
            func,  jac = self._wrap_func_and_jac(con)
            lb, ub, keep_feasible = con.get_bounds_as_array(use_norm)
            cons.append(NonlinearConstraint(
                func, lb, ub, jac, keep_feasible=keep_feasible))

        return cons

    def _wrap_callback(self):

        def callback(x: ndarray):
            self.callback()
            self.iter += 1

        return callback

    def _convert_result(self, _result: OptimizeResult) -> Dict[str, any]:
        result = {}

        result["inputs"] = array_to_dict_1d(
            self.disc.input_vars, _result.x)
        if self.disc.use_norm:
            result["inputs"] = denormalize_dict_1d(
                self.disc.input_vars, result["inputs"])

        result["objective"] = _result.fun
        result["jac"] = array_to_dict_1d(self.disc.input_vars, _result.jac)
        result["iter"] = _result.nit
        result["message"] = _result.message
        result["converged"] = _result.success

        return result

    def solve(self, input_values: Dict[str, ndarray], use_norm: bool) -> Dict[str, any]:
        # Reset iteration number
        self.iter = 0

        # Normalize the input values if needed,
        # and covert to 1d numpy array
        if use_norm:
            input_values = normalize_dict_1d(
                self.disc.input_vars, input_values)
        input_values = dict_to_array_1d(
            self.disc.input_vars, input_values)

        # Wrap the objective function, its jacobian,
        # the bounds, the constraints and the callback
        fun, jac = self._wrap_func_and_jac(self.disc.output_vars[0], True)
        bounds = self._wrap_bounds(use_norm)
        cons = self._wrap_constraints(use_norm)
        callback = self._wrap_callback()

        # Solve the optimization problem using scipy minimize
        result = minimize(fun=fun, x0=input_values, jac=jac,
                          bounds=bounds, constraints=cons, callback=callback,
                          method=self.method, tol=self.tol, options=self.options)

        result = self._convert_result(result)

        return result
