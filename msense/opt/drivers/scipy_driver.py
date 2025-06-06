from typing import Dict, Tuple

from numpy import ndarray
from scipy.optimize import NonlinearConstraint, Bounds, minimize

from msense.core.variable import Variable
from msense.core.discipline import Discipline
from msense.utils.array_and_dict_utils import concatenate_variable_bounds
from msense.utils.array_and_dict_utils import array_to_dict_1d, dict_to_array_1d
from msense.utils.array_and_dict_utils import normalize_dict_1d, denormalize_dict_1d
from msense.utils.array_and_dict_utils import array_to_dict_2d, dict_to_array_2d, denormalize_dict_2d
from msense.opt.drivers.driver import Driver


class ScipyDriver(Driver):
    """
    This Driver subclass implements the functionality needed to solve an OptProblem using SciPy optimizers.
    """

    def __init__(self, problem: Discipline, method="SLSQP", **kwargs):
        super().__init__(problem, **kwargs)
        self.method = method
        self.options = {}

    def _wrap_objective(self):
        def func(x: ndarray) -> float:
            x = array_to_dict_1d(self.disc.input_vars, x)
            obj_value = self.disc.eval(x)[self.disc.output_vars[0].name]
            if self.iter == 0:
                self._callback()
            return obj_value
        return func

    def _wrap_gradient(self):
        def gradient(x: ndarray) -> ndarray:
            x = array_to_dict_1d(self.disc.input_vars, x)
            grad = self.disc.differentiate(x)
            grad = dict_to_array_2d(self.disc.input_vars,
                                    [self.disc.output_vars[0]], grad, flatten=True)
            return grad
        return gradient

    def _wrap_constraints(self, use_norm: bool) -> NonlinearConstraint:
        def wrap_single_constraint(con: Variable):
            def constraint(x: ndarray) -> ndarray:
                x = array_to_dict_1d(self.disc.input_vars, x)
                con_value = self.disc.eval(x)[con.name]
                return con_value

            def jacobian(x: ndarray) -> ndarray:
                x = array_to_dict_1d(self.disc.input_vars, x)
                con_jac = self.disc.differentiate(x)
                con_jac = dict_to_array_2d(self.disc.input_vars,
                                           [con], con_jac)
                return con_jac

            # This is needed because regardless of whether the method
            # requires gradients, the provided constraint jacobian func
            # will be called, so it should not be provided
            if self.method in ["Powell", "Nelder-Mead",
                               "COBYLA", "COBYQA"]:
                return constraint, '2-point'
            else:
                return constraint, jacobian

        constraints = []
        for con in self.disc.output_vars[1:]:
            constraint, jacobian = wrap_single_constraint(con)
            cl, cu, kf = con.get_bounds_as_array(use_norm)
            constraints.append(NonlinearConstraint(
                constraint, cl, cu, jacobian, keep_feasible=kf[0]))
        return constraints

    def _wrap_callback(self):
        if self.method == "trust-constr":
            def callback(x, result):
                self._callback()
            return callback
        elif self.method in ["TNC", "SLSQP", "COBYLA"]:
            def callback(x):
                self._callback()
            return callback
        else:
            def callback(result):
                self._callback()
            return callback

    def _wrap_bounds(self, use_norm):
        lb, ub, kf = concatenate_variable_bounds(
            self.disc.input_vars, use_norm)
        return Bounds(lb, ub, kf)

    def solve(self, input_values: Dict[str, ndarray], use_norm: bool) -> Tuple[bool, str]:
        # Normalize the input values if needed,
        # and covert to 1d numpy array
        if use_norm:
            input_values = normalize_dict_1d(
                self.disc.input_vars, input_values)
        input_values = dict_to_array_1d(
            self.disc.input_vars, input_values)

        # Reset the iteration number
        self.iter = 0

        # Solve the optimization problem using SciPy
        self.options["maxiter"] = self.n_iter_max
        result = minimize(fun=self._wrap_objective(),
                          x0=input_values,
                          jac=self._wrap_gradient(),
                          bounds=self._wrap_bounds(use_norm),
                          constraints=self._wrap_constraints(use_norm),
                          callback=self._wrap_callback(),
                          method=self.method, tol=self.tol,
                          options=self.options)

        # Driver convergence and related message
        converged, message = result["success"], result["message"]

        return converged, message
