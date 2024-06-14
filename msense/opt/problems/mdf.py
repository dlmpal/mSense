from typing import List

from msense.core.discipline import Discipline
from msense.solver.solver import Solver
from msense.jacobians.jacobian_assembler import JacobianAssembler
from msense.opt.problems.opt_problem import OptProblem


class MDF(OptProblem):
    """
    The Multidisciplinary Feasible (MDF) approach for MDO problems.
    """

    def __init__(self, disciplines: List[Discipline], solver: Solver, **kwargs) -> None:
        self.disciplines = disciplines
        self.solver = solver
        self.assembler = JacobianAssembler()
        super().__init__(**kwargs)

    def _eval(self):
        # Update the disciplinary inputs by the values
        # provided by optimizer
        for disc in self.disciplines:
            disc.add_default_inputs(self._values)

        # Solve the system
        self.solver.solve()

        # Grab the values of the constraints and the objective
        # from the disciplinary outputs
        disc_outputs = {}
        for disc in self.disciplines:
            disc_outputs.update(disc.get_output_values())
        for var in self.output_vars:
            self._values[var.name] = disc_outputs[var.name]

    def _differentiate(self) -> None:
        # Evaluate the discipline partials
        disc_partials = {}
        for disc in self.disciplines:
            disc_partials.update(disc.differentiate())

        # Assemble the total (coupled) derivatives
        self._jac = self.assembler.assemble_total(
            self.input_vars, self.output_vars, self.solver.coupling_vars, disc_partials)
