from msense.solver.solver import Solver


class NonlinearJacobi(Solver):
    """
    This Solver sub-class implements the generalized 
    or non-linear Jacobi iteration:

    Yi^(k+1) = Yi(Xi^k),

    where Xi^k = [xi^k z^k y1i^k ... yni^k], j =/= i
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _single_iteration(self) -> None:
        outputs = {}
        for disc in self.disciplines:
            outputs.update(disc.eval(self._old_values))

        for var in self.coupling_vars:
            self._values[var.name] = outputs[var.name]
