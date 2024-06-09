from typing import List

from numpy import ones

from msense.core.constants import FLOAT_DTYPE
from msense.core.variable import Variable
from msense.core.discipline import Discipline
from msense.utils.graph_utils import get_couplings
from msense.opt.problems.opt_problem import OptProblem


class IDF(OptProblem):
    """
    The Individual Discipline Feasible (IDF) approach for MDO problems.
    """

    def __init__(self, disciplines: List[Discipline], **kwargs) -> None:
        self.disciplines = disciplines

        # Get the coupling variables and add them
        # to the design variables
        self.coupling_vars = get_couplings(self.disciplines)
        kwargs["design_vars"] += self.coupling_vars

        # Create feasibility/consistency constraints
        self.feasibility_constraints = [
            Variable(var.name + "_con", var.size, 0, 0) for var in self.coupling_vars]
        if kwargs["constraints"] is None:
            kwargs["constraints"] = self.feasibility_constraints
        else:
            kwargs["constraints"] += self.feasibility_constraints

        super().__init__(**kwargs)

    def _eval(self):
        # Evaluate the disciplines
        outputs = {}
        for disc in self.disciplines:
            inputs = {}
            for var in disc.input_vars:
                if var.name in self._values:
                    inputs[var.name] = self._values[var.name]
            outputs.update(disc.eval(inputs))

        # Evaluate feasibility constraints
        for var in self.coupling_vars:
            outputs[var.name + "_con"] = self._values[var.name] - \
                outputs[var.name]

        # Set the values of the constraints and the objective
        for var in self.output_vars:
            self._values[var.name] = outputs[var.name]

    def _differentiate(self) -> None:
        # Evaluate the partials
        partials = {}
        for disc in self.disciplines:
            partials.update(disc.differentiate())

        # Feasibility constraint jacobians
        for out_var in self.coupling_vars:
            for in_var in self.design_vars:
                if in_var.name == out_var.name:
                    self._jac[out_var.name + "_con"][out_var.name] = ones(
                        (out_var.size, out_var.size), dtype=FLOAT_DTYPE)
                elif in_var.name in partials[out_var.name]:
                    self._jac[out_var.name + "_con"][in_var.name] = - \
                        partials[out_var.name][in_var.name]

        # Objective and constraint jacobians
        for out_var in self.output_vars:
            if out_var.name in partials:
                for in_var in self.design_vars:
                    if in_var.name in partials[out_var.name]:
                        self._jac[out_var.name][in_var.name] = partials[out_var.name][in_var.name]
