from typing import Dict, List, Mapping

from numpy import ndarray
from scipy.optimize import NonlinearConstraint, Bounds, minimize

from msense.opt.formulation import Constraint
from msense.utils.array_and_dict_utils import dict_to_array_2d
from msense.opt.drivers.driver import Driver, DriverResult

#: SciPy methods that do not use gradients, and so must not be handed a jacobian.
GRADIENT_FREE_METHODS = ("Powell", "Nelder-Mead", "COBYLA", "COBYQA")


class ScipyDriver(Driver):
    """
    This Driver subclass implements the functionality needed to solve an OptProblem
    using SciPy optimizers.
    """

    supports_multi_objective = False
    supports_nonlinear_constraints = True
    supports_equality_constraints = True

    def __init__(self, problem, method: str = "SLSQP", **kwargs):
        super().__init__(problem, **kwargs)
        self.method = method

        # The values at the most recent iterate, for the history
        self._latest: Dict[str, ndarray] = None

    @property
    def requires_gradients(self) -> bool:
        return self.method not in GRADIENT_FREE_METHODS

    def _wrap_objective(self):
        def func(x: ndarray) -> float:
            values = self.evaluate(x)
            self._latest = {**self.design_dict(x), **values}
            if self.iter == 0:
                self._callback(values=self._latest)
            return self.prob.scalar_objective(values)
        return func

    def _wrap_gradient(self):
        def gradient(x: ndarray) -> ndarray:
            return self.prob.scalar_objective_jac(self.differentiate(x))
        return gradient

    def _wrap_constraints(self) -> List[NonlinearConstraint]:
        def wrap_single_constraint(con: Constraint):
            def constraint(x: ndarray) -> ndarray:
                return self.evaluate(x)[con.name]

            def jacobian(x: ndarray) -> ndarray:
                return dict_to_array_2d(self.prob.design_vars, [con.var],
                                        self.differentiate(x))

            # Regardless of whether the method uses gradients, the provided
            # constraint jacobian is called, so it must not be provided for the
            # gradient-free methods.
            if self.requires_gradients:
                return constraint, jacobian
            else:
                return constraint, '2-point'

        constraints = []
        for con in self.prob.constraints:
            func, jac = wrap_single_constraint(con)
            # SciPy takes the two-sided constraint form directly, so the bounds
            # of the constraint variable are used as they are.
            cl, cu, kf = con.var.get_bounds_as_array(use_normalization=False)
            constraints.append(NonlinearConstraint(func, cl, cu, jac, keep_feasible=kf[0]))
        return constraints

    def _wrap_callback(self):
        if self.method == "trust-constr":
            def callback(x, result):
                self._callback(values=self._latest)
            return callback
        elif self.method in ["TNC", "SLSQP", "COBYLA"]:
            def callback(x):
                self._callback(values=self._latest)
            return callback
        else:
            def callback(result):
                self._callback(values=self._latest)
            return callback

    def _wrap_bounds(self, use_norm: bool) -> Bounds:
        xl, xu = self.prob.design_bounds(use_norm)
        return Bounds(xl, xu)

    def solve(self, x0: Mapping[str, ndarray], use_norm: bool) -> DriverResult:

        # Convert from dict to array and normalize
        x0 = self.starting_array(x0)

        # Reset iteration counter
        self.iter = 0

        self.options["maxiter"] = self.n_iter_max
        result = minimize(fun=self._wrap_objective(),
                          x0=x0,
                          jac=self._wrap_gradient() if self.requires_gradients else None,
                          bounds=self._wrap_bounds(use_norm),
                          constraints=self._wrap_constraints(),
                          callback=self._wrap_callback(),
                          method=self.method, tol=self.tol,
                          options=self.options)

        x = result["x"]
        return DriverResult(converged=bool(result["success"]),
                            message=str(result["message"]),
                            x=self.design_dict(x),
                            values=self.evaluate(x),
                            n_eval=self.prob.n_eval,
                            n_iter=self.iter)
