"""
The Multidisciplinary Feasible (MDF) formulation.

The design variables are pushed into the disciplines' default inputs, the MDA
solver drives the coupling variables to consistency, and the objective and
constraints are read back. Because that push/solve/pull sequence mutates the
disciplines, an MDF problem is **not safe to evaluate concurrently** -- do not
give it to a driver using a parallel eval_batch.
"""
from typing import Dict, List, Mapping

from numpy import ndarray

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

    def _eval(self, inputs: Mapping[str, ndarray]) -> Dict[str, ndarray]:
        # Update the disciplinary inputs by the values provided by driver
        for disc in self.disciplines:
            disc.add_default_inputs(inputs)

        # Solve the system
        couplings = self.solver.solve()

        # Grab the values of the constraints and the objective(s)
        # from the disciplinary outputs
        outputs = {}
        for disc in self.disciplines:
            disc.add_default_inputs(couplings)
            outputs |= disc.eval()

        return outputs

    def _differentiate(self, inputs: Mapping[str, ndarray],
                       outputs: Mapping[str, ndarray]) -> Dict[str, Dict[str, ndarray]]:
        self._eval(inputs)

        # Evaluate the discipline partials
        disc_partials = {}
        for disc in self.disciplines:
            disc_partials.update(disc.differentiate())

        # Assemble the total (coupled) derivatives
        return self.assembler.assemble_total(
            self.input_vars, self.output_vars, self.solver.coupling_vars, disc_partials)
