from typing import Dict, List

from numpy import ndarray, ones
from numpy.linalg import norm

from msense.core.constants import FLOAT_DTYPE
from msense.core.variable import Variable
from msense.core.discipline import Discipline
from msense.utils.graph_utils import get_couplings
from msense.opt.problems.opt_problem import OptProblem


class IDF(OptProblem):
    """
    The Individual Discipline Feasible (IDF) approach for MDO problems.
    """

    def __init__(self, disciplines: List[Discipline], scalar_feasibility_constraints: bool = False,
                 feasibility_tol: float = 0.0, **kwargs) -> None:
        self.disciplines = disciplines
        self.scalar_feasiblity_constraints = scalar_feasibility_constraints

        # Get the coupling variables and add them
        # to the design variables
        self.coupling_vars = get_couplings(self.disciplines)
        kwargs["design_vars"] += self.coupling_vars

        # Create feasibility/consistency constraints
        self.feasibility_constraints = []
        for var in self.coupling_vars:
            con_size = 1 if self.scalar_feasiblity_constraints else var.size
            self.feasibility_constraints.append(Variable(var.name + "_con", con_size, lb=-
                                                         feasibility_tol, ub=feasibility_tol))
        if kwargs["constraints"] is None:
            kwargs["constraints"] = self.feasibility_constraints
        else:
            kwargs["constraints"] += self.feasibility_constraints

        super().__init__(**kwargs)

    def _compute_feasibility_constraint(self, var: Variable, disc_outputs: Dict[str, ndarray]) -> ndarray:
        con_value = None
        if self.scalar_feasiblity_constraints:
            con_value = norm(
                self._values[var.name] - disc_outputs[var.name])**2
        else:
            con_value = self._values[var.name] - disc_outputs[var.name]
        return con_value

    def _compute_feasibility_constraint_jac(self, in_var: Variable, out_var: Variable,
                                            disc_outputs: Dict[str, ndarray], disc_partials: Dict[str, Dict[str, ndarray]]) -> ndarray:
        con_jac = None
        if self.scalar_feasiblity_constraints:
            if in_var.name == out_var.name:
                con_jac = 2 * (self._values[out_var.name] -
                               disc_outputs[out_var.name])
            elif in_var.name in disc_partials[out_var.name]:
                con_jac = -2 * (self._values[out_var.name] - disc_outputs[out_var.name]
                                ) @ disc_partials[out_var.name][in_var.name]
        else:
            if in_var.name == out_var.name:
                con_jac = ones((out_var.size, out_var.size), FLOAT_DTYPE)
            elif in_var.name in disc_partials[out_var.name]:
                con_jac = -disc_partials[out_var.name][in_var.name]
        return con_jac

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
                    "_con"] = self._compute_feasibility_constraint(var, outputs)

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
                con_jac = self._compute_feasibility_constraint_jac(
                    in_var, out_var, outputs, partials)
                if con_jac is not None:
                    self._jac[out_var.name + "_con"][in_var.name] = con_jac

        # Objective and constraint jacobians
        for out_var in self.output_vars:
            if out_var.name in partials:
                for in_var in self.design_vars:
                    if in_var.name in partials[out_var.name]:
                        self._jac[out_var.name][in_var.name] = partials[out_var.name][in_var.name]
