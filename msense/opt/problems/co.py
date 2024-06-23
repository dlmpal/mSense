from typing import Dict, List, Tuple

from numpy import ndarray
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
                 objective: Variable):

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

        super().__init__(discipline.name + "SubProblem",
                         self.local_design_vars + self.global_design_vars,
                         objective,
                         self.constraints)

        # Values provided by the system-level optimizer
        # These are set at the beginning
        # of each system-level optimizer iteration
        self._global_values = {}

    def set_global_values(self, values: Dict[str, ndarray]):
        self._global_values = copy_dict_1d(
            self.local_design_vars + self.global_design_vars +
            self.input_couplings + self.output_couplings, values)

    def _eval(self):
        # Gather the discipline inputs
        local_values = {}
        for var in self.design_vars:
            local_values[var.name] = self._values[var.name].copy()
        for var in self.input_couplings:
            local_values[var.name] = self._global_values[var.name].copy()

        # Evaluate the discipline
        local_values.update(self.disc.eval(local_values))

        # Update the local constraints
        for var in self.constraints:
            self._values[var.name] = local_values[var.name]

        # Evaluate the feasibility constraint
        self._values[self.objective.name] = zeros(
            self.objective.size, FLOAT_DTYPE)

        for var in self.local_design_vars:
            self._values[self.objective.name] += norm(
                self._global_values[var.name] - self._values[var.name])**2

        for var in self.global_design_vars:
            self._values[self.objective.name] += norm(
                self._values[var.name] - self._global_values[var.name])**2

        for var in self.output_couplings:
            self._values[self.objective.name] += norm(
                self._global_values[var.name] - local_values[var.name])**2

    def _differentiate(self) -> None:

        disc_values = self.disc.get_values()
        disc_jac = self.disc.differentiate()

        for in_var in self.local_design_vars:
            self._jac[self.objective.name][in_var.name] = -2 * \
                (self._global_values[in_var.name] - self._values[in_var.name])

            for out_var in self.output_couplings:
                self._jac[self.objective.name][in_var.name] += -2 * \
                    (self._global_values[out_var.name] - disc_values[out_var.name]
                     ) @ disc_jac[out_var.name][in_var.name]

        for in_var in self.global_design_vars:
            self._jac[self.objective.name][in_var.name] = 2 * \
                (self._values[in_var.name] - self._global_values[in_var.name])

            for out_var in self.output_couplings:
                self._jac[self.objective.name][in_var.name] += -2 * \
                    (self._global_values[out_var.name] - disc_values[out_var.name]
                     ) @ disc_jac[out_var.name][in_var.name]

        for con in self.constraints:
            for in_var in self.design_vars:
                self._jac[con.name][in_var.name] = disc_jac[con.name][in_var.name]


class CO(OptProblem):
    """
    The Collaborative Optimization (CO) approach for MDO problems.
    """

    def __init__(self, disciplines: List[Discipline], warm_start: bool = False,
                 feasibility_tol: float = 0.0, **kwargs) -> None:
        self.disciplines = disciplines[:-1]
        self.objective_discipline = disciplines[-1]
        self.warm_start = warm_start

        # Get the coupling variables
        self.coupling_vars = get_couplings(self.disciplines)

        # Separate the local and global design variables
        design_vars = kwargs["design_vars"]
        self.local_design_vars, self.global_design_vars = separate_local_global(
            design_vars, self.disciplines)

        # Separate the local and global constraints
        if kwargs["constraints"] is None:
            kwargs["constraints"] = []
        self.global_constraints = []
        for var in kwargs["constraints"]:
            if var in self.objective_discipline.output_vars:
                self.global_constraints.append(var)

        # Create the subproblems and
        # the disciplinary feasibility constraints
        self.subproblems: List[COSubProblem] = []
        self.feasibility_constraints = []
        for disc in self.disciplines:
            con = Variable(disc.name + "_con",
                           lb=-feasibility_tol, ub=feasibility_tol)
            subprob = COSubProblem(disc,
                                   self.local_design_vars[disc],
                                   self.global_design_vars,
                                   self.coupling_vars,
                                   kwargs["constraints"],
                                   con)
            self.subproblems.append(subprob)
            self.feasibility_constraints.append(con)

        # Initialize the base OptProblem
        kwargs["design_vars"] += self.coupling_vars
        kwargs["constraints"] = self.global_constraints + \
            self.feasibility_constraints
        super().__init__(**kwargs)

    def _eval(self) -> None:
        # Solve the subproblems
        # Get the values of the feasibility constraints
        for subprob in self.subproblems:
            subprob.set_global_values(self._values)
            if self.driver.iter == 0 or self.warm_start is False:
                subprob.solve(copy_dict_1d(subprob.design_vars, self._values))
            else:
                subprob.solve(subprob.get_input_values())
            self._values[subprob.objective.name] = subprob.get_output_values()[
                subprob.objective.name]

        # Evaluate the objective and global constraints
        obj_disc_values = self.objective_discipline.eval(self._values)
        self._values[self.objective.name] = obj_disc_values[self.objective.name]
        for var in self.global_constraints:
            self._values[var.name] = obj_disc_values[var.name]

    def _differentiate(self) -> None:
        # Differentiate the feasibility constraints
        # This is done according to the post-optimal sensitivity results
        # of Braun
        for subprob in self.subproblems:
            values = subprob._global_values
            opt_values = subprob.disc.get_values()

            for var in subprob.local_design_vars:
                self._jac[subprob.objective.name][var.name] = -2 * \
                    (values[var.name] - opt_values[var.name])

            for var in subprob.global_design_vars:
                self._jac[subprob.objective.name][var.name] = 2 * \
                    (values[var.name] - opt_values[var.name])

            for var in subprob.output_couplings:
                self._jac[subprob.objective.name][var.name] = 2 * \
                    (values[var.name] - opt_values[var.name])

        # Differentiate the objective discipline
        obj_disc_jac = self.objective_discipline.differentiate()
        for out_var in self.objective_discipline.output_vars:
            for in_var in self.objective_discipline.input_vars:
                self._jac[out_var.name][in_var.name] = obj_disc_jac[out_var.name][in_var.name]
