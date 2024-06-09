from typing import List

from numpy.linalg import norm

from msense.core.variable import Variable
from msense.core.discipline import Discipline
from msense.utils.graph_utils import get_couplings
from msense.opt.problems.opt_problem import OptProblem


class IDF(OptProblem):
    """
    The Individual Discipline Feasible (IDF) approach for MDO problems.
    """

    def __init__(self, disciplines: List[Discipline], feasibility_tol: float = 1e-6, **kwargs) -> None:
        self.disciplines = disciplines

        # Get the coupling variables and add them
        # to the design variables
        self.coupling_vars = get_couplings(self.disciplines)
        kwargs["design_vars"] += self.coupling_vars

        # Create feasibility/consistency constraints
        self.feasibility_constraints = [
            Variable(var.name + "_con", lb=0, ub=feasibility_tol) for var in self.coupling_vars]
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
            outputs[var.name +
                    "_con"] = norm(self._values[var.name] - outputs[var.name])**2

        # Set the values of the constraints and the objective
        for var in self.output_vars:
            self._values[var.name] = outputs[var.name]

    def _differentiate(self) -> None:
        # Evaluate the partials
        partials = {}
        outputs = {}
        for disc in self.disciplines:
            outputs.update(disc.get_output_values())
            partials.update(disc.differentiate())

        # Feasibility constraint jacobians
        for out_var in self.coupling_vars:
            for in_var in self.design_vars:
                if in_var.name == out_var.name:
                    self._jac[out_var.name + "_con"][out_var.name] = 2 * \
                        (self._values[out_var.name] - outputs[out_var.name])
                elif in_var.name in partials[out_var.name]:
                    self._jac[out_var.name + "_con"][in_var.name] = -2 * \
                        (self._values[out_var.name] - outputs[out_var.name]
                         )  @ partials[out_var.name][in_var.name]

        # Objective and constraint jacobians
        for out_var in self.output_vars:
            if out_var.name in partials:
                for in_var in self.design_vars:
                    if in_var.name in partials[out_var.name]:
                        self._jac[out_var.name][in_var.name] = partials[out_var.name][in_var.name]
