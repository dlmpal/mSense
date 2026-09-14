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


#
# Coupled systems, for the MDA solvers
#

y1 = Variable("y1")
y2 = Variable("y2")


class LinearCoupling1(Discipline):
    """
    y1 = a - b*y2.

    Paired with LinearCoupling2 this has the exact fixed point
    y1 = (a - b*c)/(1 + b*d), y2 = c + d*y1, which a Newton step reaches
    in one iteration because the system is linear.
    """

    def __init__(self, a: float = 3.0, b: float = 0.2):
        self.a, self.b = a, b
        super().__init__("Linear1", [y2], [y1], cache_type=None)

    def _eval(self, inputs):
        return {"y1": self.a - self.b * inputs["y2"]}

    def _differentiate(self, inputs, outputs):
        return {"y1": {"y2": np.atleast_2d(-self.b)}}


class LinearCoupling2(Discipline):
    """
    y2 = c + d*y1.
    """

    def __init__(self, c: float = 2.0, d: float = 0.5):
        self.c, self.d = c, d
        super().__init__("Linear2", [y1], [y2], cache_type=None)

    def _eval(self, inputs):
        return {"y2": self.c + self.d * inputs["y1"]}

    def _differentiate(self, inputs, outputs):
        return {"y2": {"y1": np.atleast_2d(self.d)}}


def linear_coupling_solution(a=3.0, b=0.2, c=2.0, d=0.5):
    """
    The exact fixed point of the linear pair above.
    """
    y1_star = (a - b * c) / (1.0 + b * d)
    return y1_star, c + d * y1_star


class SellarCoupling1(Discipline):
    """
    y1 = z1^2 + z2 + x1 - 0.2*y2, the first Sellar coupling.
    """

    def __init__(self, z1=1.0, z2=1.0, x1=1.0):
        self.z1, self.z2, self.x1 = z1, z2, x1
        super().__init__("Sellar1", [y2], [y1], cache_type=None)

    def _eval(self, inputs):
        return {"y1": self.z1**2 + self.z2 + self.x1 - 0.2 * inputs["y2"]}

    def _differentiate(self, inputs, outputs):
        return {"y1": {"y2": np.atleast_2d(-0.2)}}


class SellarCoupling2(Discipline):
    """
    y2 = sqrt(y1) + z1 + z2, the second Sellar coupling.
    """

    def __init__(self, z1=1.0, z2=1.0):
        self.z1, self.z2 = z1, z2
        super().__init__("Sellar2", [y1], [y2], cache_type=None)

    def _eval(self, inputs):
        return {"y2": np.sqrt(np.abs(inputs["y1"])) + self.z1 + self.z2}

    def _differentiate(self, inputs, outputs):
        return {"y2": {"y1": np.atleast_2d(
            0.5 / np.sqrt(np.abs(inputs["y1"])))}}


def sellar_coupling_solution(z1=1.0, z2=1.0, x1=1.0):
    """
    The exact fixed point of the Sellar pair.

    Substituting gives u^2 + 0.2u - (z1^2 + z2 + x1 - 0.2(z1+z2)) = 0 with
    u = sqrt(y1); the positive root is taken.
    """
    k = z1**2 + z2 + x1 - 0.2 * (z1 + z2)
    u = 0.5 * (-0.2 + np.sqrt(0.04 + 4.0 * k))
    return u**2, u + z1 + z2
