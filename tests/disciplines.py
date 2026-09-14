"""
Test disciplines and problem builders shared across the test modules.
"""
import numpy as np

from msense.api import CachePolicy, Discipline, EvaluationFailure
from msense.api import Objective, Variable, create_opt_problem

# min x1^2 + x2^2 over the box gives (2, 3), y = 13.
# With g = x1 + x2 >= 10 it gives (5, 5), y = 50.
x1 = Variable("x1", lb=2.0, ub=100.0)
x2 = Variable("x2", lb=3.0, ub=90.0)
y = Variable("y")
g = Variable("g", lb=10.0)


class Parabola(Discipline):
    """
    y = x1^2 + x2^2,  g = x1 + x2, with analytic partials.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("cache_policy", CachePolicy.FULL)
        super().__init__("Parabola", [x1, x2], [y, g], **kwargs)

    def _eval(self, inputs):
        return {"y": inputs["x1"]**2 + inputs["x2"]**2,
                "g": inputs["x1"] + inputs["x2"]}

    def _differentiate(self, inputs, outputs):
        return {"y": {"x1": np.atleast_2d(2 * inputs["x1"]),
                      "x2": np.atleast_2d(2 * inputs["x2"])},
                "g": {"x1": np.atleast_2d(1.0),
                      "x2": np.atleast_2d(1.0)}}


class OutputDependent(Discipline):
    """
    z = y^2, where y is this discipline's own output.

    Its partial dz/dx1 = 2*y*dy/dx1 needs the output value, which is what makes
    it the test of the eval-before-differentiate contract.
    """

    def __init__(self):
        self.z = Variable("z")
        super().__init__("OutputDependent", [x1], [y, self.z])

    def _eval(self, inputs):
        y_val = inputs["x1"]**2
        return {"y": y_val, "z": y_val**2}

    def _differentiate(self, inputs, outputs):
        # Deliberately expressed through outputs, not recomputed from inputs
        return {"y": {"x1": np.atleast_2d(2 * inputs["x1"])},
                "z": {"x1": np.atleast_2d(2 * outputs["y"] * 2 * inputs["x1"])}}


class Flaky(Discipline):
    """
    Fails whenever x1 > 50, standing in for a solver that will not converge on
    part of the design box.
    """

    def __init__(self):
        super().__init__("Flaky", [x1, x2], [y, g])

    def _eval(self, inputs):
        if inputs["x1"][0] > 50.0:
            raise EvaluationFailure(f"no solution for x1={inputs['x1'][0]}")
        return {"y": inputs["x1"]**2 + inputs["x2"]**2,
                "g": inputs["x1"] + inputs["x2"]}


def make_problem(disc=None, objectives=None, constraints=None, **kwargs):
    """
    Build a single-discipline problem over the parabola.
    """
    disc = disc if disc is not None else Parabola()
    objectives = objectives if objectives is not None else [Objective(y)]
    return disc, create_opt_problem("single_discipline", [disc],
                                    design_vars=[x1, x2],
                                    objectives=objectives,
                                    constraints=constraints,
                                    name="ParabolaOpt", **kwargs)
