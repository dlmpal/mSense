from scipy.sparse import eye
from scipy.sparse.linalg import spsolve

from msense.utils.array_and_dict_utils import array_to_dict_1d, dict_to_array_1d
from msense.jacobians.jacobian_assembler import JacobianAssembler
from msense.solver.solver import Solver


class NewtonRaphson(Solver):
    """
    This Solver sub-class uses the Newton-Raphson iteration 
    for a system of non-linear equations:

    (dR/dY)^k * Ycorr^k = R^k

    Y^(k+1) = Y^k + Ycorr^k
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.assembler = JacobianAssembler()

    def _single_iteration(self) -> None:
        # Evaluate and differentiate the disciplines
        outputs, partials = {}, {}
        for disc in self.disciplines:
            outputs.update(disc.eval(self._old_values))
            partials.update(disc.differentiate(self._old_values))

        # Compute correction
        R = {}
        for var in self.coupling_vars:
            R[var.name] = self._old_values[var.name] - outputs[var.name]
        R = dict_to_array_1d(self.coupling_vars, R)
        dRdy = self.assembler.assemble_partial(
            self.coupling_vars, self.coupling_vars, partials, True)
        corr = spsolve(dRdy.tocsr(), -R)
        corr = array_to_dict_1d(self.coupling_vars, corr)

        # Update values
        for var in self.coupling_vars:
            self._values[var.name] = self._old_values[var.name] + \
                corr[var.name]
