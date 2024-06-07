from typing import Dict, List, Tuple

from numpy import zeros
from numpy.linalg import norm

from msense.core.constants import FLOAT_DTYPE
from msense.core.variable import Variable
from msense.core.discipline import Discipline
from msense.utils.array_and_dict_utils import copy_dict_1d
from msense.utils.graph_utils import get_couplings
from msense.utils.graph_utils import separate_local_global
from msense.opt.problems.opt_problem import OptProblem


class COSubProblem(OptProblem):
    """
    Optimization subproblem for the i-th discipline, for an MDO problem solved by CO.
    """

    def __init__(self, discipline: Discipline,
                 local_design_vars: List[Variable],
                 global_design_vars: List[Variable],
                 coupling_vars: List[Variable],
                 constraints: List[Variable],
                 J: Variable):

        self.disc = discipline

        self.local_design_vars = local_design_vars

        self.global_design_vars = []
        for var in global_design_vars:
            if var in self.disc.input_vars:
                self.global_design_vars.append(var)

        self.output_couplings = []
        self.input_couplings = []
        for var in coupling_vars:
            if var in self.disc.output_vars:
                self.output_couplings.append(var)
            elif var in self.disc.input_vars:
                self.input_couplings.append(var)

        self.constraints = []
        for var in constraints:
            if var in self.disc.output_vars:
                self.constraints.append(var)

        # Consistency constraint
        # This is the objective for the local optimization problem
        self.J = J

        # Values provided by the system-level optimizer
        self.global_values = {}

        super().__init__(discipline.name + "SubProblem",
                         self.local_design_vars + self.global_design_vars,
                         self.J,
                         self.constraints)

    def _eval(self):
        # Gather the discipline inputs
        disc_input_values = {}
        for var in self.design_vars:
            disc_input_values[var.name] = self._values[var.name].copy()
        for var in self.input_couplings:
            disc_input_values[var.name] = self.global_values[var.name].copy()

        # Evaluate the discipline
        disc_output_values = self.disc.eval(disc_input_values)

        # Update the local constraints
        for var in self.constraints:
            self._values[var.name] = disc_output_values[var.name]

        # Evaluate the consistency constraint J
        self._values[self.J.name] = zeros(self.J.size, FLOAT_DTYPE)

        for var in self.local_design_vars:
            self._values[self.J.name] += norm(
                self.global_values[var.name] - self._values[var.name])**2

        for var in self.global_design_vars:
            self._values[self.J.name] += norm(
                self._values[var.name] - self.global_values[var.name])**2

        for var in self.output_couplings:
            self._values[self.J.name] += norm(
                self.global_values[var.name] - disc_output_values[var.name])**2

    def _differentiate(self) -> None:

        disc_output_values = self.disc.get_output_values()
        disc_jac = self.disc.differentiate()

        for var in self.local_design_vars:
            self._jac[self.J.name][var.name] = -2 * \
                (self.global_values[var.name] - self._values[var.name])
            for cvar in self.output_couplings:
                self._jac[self.J.name][var.name] += -2 * \
                    (self.global_values[cvar.name] - disc_output_values[cvar.name]
                     ) @ disc_jac[cvar.name][var.name]

        for var in self.global_design_vars:
            self._jac[self.J.name][var.name] = 2 * \
                (self._values[var.name] - self.global_values[var.name])
            for cvar in self.output_couplings:
                self._jac[self.J.name][var.name] += -2 * \
                    (self.global_values[cvar.name] - disc_output_values[cvar.name]
                     ) @ disc_jac[cvar.name][var.name]

        for con in self.constraints:
            for var in self.design_vars:
                self._jac[con.name][var.name] = disc_jac[con.name][var.name]


class CO(OptProblem):
    """
    The Collaborative Optimization (CO) approach for MDO problems.
    """

    def __init__(self, disciplines: List[Discipline], **kwargs) -> None:
        self.disciplines = disciplines[:-1]
        self.objective_discipline = disciplines[-1]

        # Get the coupling variables
        self.coupling_vars = get_couplings(self.disciplines)

        # Separate the local and global constraints
        if kwargs["constraints"] is None:
            kwargs["constraints"] = []
        self.global_constraints = []  # Global constraints
        for var in kwargs["constraints"]:
            if var in self.objective_discipline.output_vars:
                self.global_constraints.append(var)

        # Separate the local and global design variables
        design_vars = kwargs["design_vars"]
        self.local_design_vars, self.global_design_vars = separate_local_global(
            design_vars, self.disciplines)

        # Create the subproblems and the feasibility constraints J
        self.subproblems: List[COSubProblem] = []
        self.feasibility_constraints = []
        for disc in self.disciplines:
            J = Variable("J_" + disc.name, lb=0, ub=0)
            subprob = COSubProblem(disc,
                                   self.local_design_vars[disc],
                                   self.global_design_vars,
                                   self.coupling_vars,
                                   kwargs["constraints"],
                                   J)
            self.subproblems.append(subprob)
            self.feasibility_constraints.append(J)

        # Initialize the base OptProblem
        kwargs["design_vars"] += self.coupling_vars
        kwargs["constraints"] = self.global_constraints + \
            self.feasibility_constraints
        super().__init__(**kwargs)

    def _eval(self) -> None:

        # Solve the subproblems
        # Get the values of the consistency constraints
        for subprob in self.subproblems:
            subprob.global_values = copy_dict_1d(
                subprob.local_design_vars +
                subprob.global_design_vars +
                subprob.input_couplings +
                subprob.output_couplings, self._values)

            subprob_design_vec = copy_dict_1d(
                subprob.local_design_vars + subprob.global_design_vars, self._values)

            subprob.solve(subprob_design_vec)

            self._values[subprob.objective.name] = subprob.get_output_values()[
                subprob.objective.name]

        # Evaluate the objective and global constraints
        obj_disc_values = self.objective_discipline.eval(self._values)
        self._values[self.objective.name] = obj_disc_values[self.objective.name]
        for var in self.global_constraints:
            self._values[var.name] = obj_disc_values[var.name]
