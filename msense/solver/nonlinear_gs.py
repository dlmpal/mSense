from msense.solver.solver import Solver


class NonlinearGS(Solver):
    """
    This Solver sub-class implements the generalized 
    or non-linear Gauss-Seidel iteration:

    Yi^(k+1) = Yi(Xi^k),

    where Xi^k = [xi^(k+1) z^(k+1) y1i^(k+1) ... y(i-1)i^(k+1)  y(i+1)i^k yni^k]
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _single_iteration(self):
        outputs = {}
        for disc in self.disciplines:
            inputs = {}
            for var in disc.input_vars:
                if var.name in outputs:
                    inputs[var.name] = outputs[var.name]
                elif var.name in self._old_values:
                    inputs[var.name] = self._old_values[var.name]
            outputs.update(disc.eval(inputs))

        for var in self.coupling_vars:
            self._values[var.name] = outputs[var.name]
