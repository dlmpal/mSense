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

        # Get the coupling variables
        self.coupling_vars = get_couplings(self.disciplines)

        # Create target variables
        self.target_vars = [Variable(c.name + "_t", c.size, c.lb, c.ub)
                            for c in self.coupling_vars]
        kwargs["design_vars"] += self.target_vars

        # Create consistency constraints
        self.target_cons = [
            Variable(c.name + "_tc", c.size, 0, 0, False) for c in self.coupling_vars]
        if kwargs["constraints"] is None:
            kwargs["constraints"] = self.target_cons
        else:
            kwargs["constraints"] += self.target_cons

        super().__init__(**kwargs)

    def _eval(self):
        # Evaluate the disciplines with the target variables values
        outputs = {}
        for disc in self.disciplines:
            inputs = {}
            for var in disc.input_vars:
                if var.name in self._values:
                    inputs[var.name] = self._values[var.name]
                elif var.name + "_t" in self._values:
                    inputs[var.name] = self._values[var.name + "_t"]
            outputs.update(disc.eval(inputs))

        # Evaluate consistency constraints
        for var in self.coupling_vars:
            outputs[var.name + "_tc"] = self._values[var.name +
                                                     "_t"] - outputs[var.name]

        # Grab the values of the constraints and the objective
        for var in self.output_vars:
            self._values[var.name] = outputs[var.name]

    def _differentiate(self) -> None:
        # Evaluate the partials
        partials = {}
        for disc in self.disciplines:
            partials.update(disc.differentiate())

        for out_var in self.doutput_vars:
            if out_var in self.target_cons:
                # Consistency constraint derivatives
                for in_var in self.dinput_vars:
                    if in_var in self.target_vars:
                        if out_var.name[:-1] == in_var.name:
                            self._jac[out_var.name][in_var.name] = ones((out_var.size, out_var.size),
                                                                        dtype=FLOAT_DTYPE)
                        elif in_var.name[:-2] in partials[out_var.name[:-3]]:
                            self._jac[out_var.name][in_var.name] = -partials[out_var.name[:-3]
                                                                             ][in_var.name[:-2]]
                    elif in_var.name in partials[out_var.name[:-3]]:
                        self._jac[out_var.name][in_var.name] = - \
                            partials[out_var.name[:-3]][in_var.name]

            else:
                # Objective and other constraint derivatives
                for in_var in self.dinput_vars:
                    if in_var in self.target_vars and in_var.name[:-2] in partials[out_var.name]:
                        self._jac[out_var.name][in_var.name] = partials[out_var.name][in_var.name[:-2]]
                    if in_var.name in partials[out_var.name]:
                        self._jac[out_var.name][in_var.name] = partials[out_var.name][in_var.name]
