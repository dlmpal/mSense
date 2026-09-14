"""
The Multidisciplinary Feasible formulation, on the Sellar problem.

Sellar has a published optimum, so the assertions are against a known answer:
z1 = 1.9776, z2 = 0, x1 = 0, f = 3.18339, with g1 active at zero.
"""
import numpy as np
import pytest

from msense.api import CachePolicy, Discipline, SolverType, Variable
from msense.api import create_driver, create_opt_problem, create_solver

# Design variables
z1 = Variable("z1", lb=-10, ub=10)
z2 = Variable("z2", lb=0, ub=10)
x1 = Variable("x1", lb=0, ub=10)

# Couplings
y1 = Variable("y1", lb=-100, ub=100)
y2 = Variable("y2", lb=-100, ub=100)

# Objective and constraints
g1 = Variable("g1", ub=0)
g2 = Variable("g2", ub=0)
f = Variable("f")

START = {"x1": 1.0, "z1": 4.0, "z2": 3.0, "y1": 0.8, "y2": 0.9}

#: Published optimum of the Sellar problem.
OPTIMUM = {"x1": 0.0, "z1": 1.9776, "z2": 0.0}
OPTIMAL_F = 3.18339


class Sellar1(Discipline):
    """y1 = sqrt(z1^2 + z2 + x1 - 0.2 y2),  g1 = 3.16 - y1^2."""

    def __init__(self):
        super().__init__("Sellar1", [z1, z2, x1, y2], [y1, g1])

    def _eval(self, inputs):
        value = np.sqrt(inputs["z1"]**2 + inputs["z2"] + inputs["x1"]
                        - 0.2 * inputs["y2"])
        return {"y1": value, "g1": 3.16 - value**2}

    def _differentiate(self, inputs, outputs):
        v = outputs["y1"]
        d = {"z1": inputs["z1"] / v, "z2": 1 / (2 * v),
             "x1": 1 / (2 * v), "y2": -0.2 / (2 * v)}
        return {"y1": dict(d),
                "g1": {k: -2 * v * dk for k, dk in d.items()}}


class Sellar2(Discipline):
    """y2 = |y1| + z1 + z2,  g2 = y2 - 24."""

    def __init__(self):
        super().__init__("Sellar2", [z1, z2, y1], [y2, g2])

    def _eval(self, inputs):
        value = np.abs(inputs["y1"]) + inputs["z1"] + inputs["z2"]
        return {"y2": value, "g2": value - 24}

    def _differentiate(self, inputs, outputs):
        sign = np.sign(inputs["y1"])
        return {"y2": {"y1": sign, "z1": 1.0, "z2": 1.0},
                "g2": {"y1": sign, "z1": 1.0, "z2": 1.0}}


class SellarObjective(Discipline):
    """f = x1^2 + z2 + y1^2 + exp(-y2)."""

    def __init__(self):
        super().__init__("Objective", [x1, z2, y1, y2], [f])

    def _eval(self, inputs):
        return {"f": inputs["x1"]**2 + inputs["z2"] + inputs["y1"]**2
                + np.exp(-inputs["y2"])}

    def _differentiate(self, inputs, outputs):
        return {"f": {"x1": 2 * inputs["x1"], "z2": 1.0,
                      "y1": 2 * inputs["y1"],
                      "y2": -np.exp(-inputs["y2"])}}


def build(cache_policy=CachePolicy.LATEST, solver_type=SolverType.NONLINEAR_GS):
    """
    A fresh MDF problem. Fresh disciplines each time: MDF pushes into their
    default inputs, so sharing them between problems would couple the two.
    """
    disciplines = [Sellar1(), Sellar2(), SellarObjective()]
    for disc in disciplines:
        disc.add_default_inputs(START)

    solver = create_solver(disciplines, solver_type, n_iter_max=30, tol=1e-12)

    return create_opt_problem("mdf", disciplines, [x1, z1, z2], f, [g1, g2],
                              name="SellarMDF", solver=solver,
                              cache_policy=cache_policy)


#
# It is available at all
#

def test_mdf_is_no_longer_refused():
    assert build() is not None


def test_idf_and_co_are_still_refused():
    with pytest.raises(NotImplementedError, match="stateless"):
        create_opt_problem("idf", [], [], objectives=[])
    with pytest.raises(NotImplementedError, match="stateless"):
        create_opt_problem("co", [], [], objectives=[])


#
# It finds the published optimum
#

def test_sellar_optimum_is_found():
    prob = build()
    prob.driver = create_driver(prob, method="SLSQP", n_iter_max=50, tol=1e-8)

    result = prob.solve(dict(START))

    assert result.converged, result.message
    for name, expected in OPTIMUM.items():
        assert np.isclose(result.x[name][0], expected, atol=1e-3), \
            f"{name} = {result.x[name][0]}, expected {expected}"
    assert np.isclose(result.values["f"][0], OPTIMAL_F, atol=1e-3)


def test_the_first_constraint_is_active_at_the_optimum():
    """
    g1 = 3.16 - y1^2 binds at the Sellar optimum; g2 does not.
    """
    prob = build()
    prob.driver = create_driver(prob, method="SLSQP", n_iter_max=50, tol=1e-8)

    result = prob.solve(dict(START))

    assert abs(result.values["g1"][0]) < 1e-4, result.values["g1"]
    assert result.values["g2"][0] < -1.0, result.values["g2"]


@pytest.mark.parametrize("solver_type", [SolverType.NONLINEAR_GS,
                                         SolverType.NONLINEAR_JACOBI,
                                         SolverType.NEWTON_RAPHSON])
def test_the_answer_does_not_depend_on_the_mda_solver(solver_type):
    prob = build(solver_type=solver_type)
    prob.driver = create_driver(prob, method="SLSQP", n_iter_max=50, tol=1e-8)

    result = prob.solve(dict(START))

    assert np.isclose(result.values["f"][0], OPTIMAL_F, atol=1e-3), solver_type


#
# The coupled derivatives
#

def test_total_derivatives_match_finite_differences():
    """
    MDF's jacobian is the *total* derivative through the converged MDA, which
    the assembler builds from the disciplinary partials. Finite-differencing
    the whole problem is an independent route to the same number.
    """
    point = {"x1": np.array([1.0]), "z1": np.array([3.0]), "z2": np.array([2.0])}

    analytic = build().differentiate({k: v.copy() for k, v in point.items()})

    step = 1e-6
    for name in ("x1", "z1", "z2"):
        forward, backward = dict(point), dict(point)
        forward[name] = point[name] + step
        backward[name] = point[name] - step

        # A fresh problem per evaluation: the MDA warm-starts from whatever the
        # disciplines currently hold, so re-using one would couple the samples.
        f_plus = build().eval({k: v.copy() for k, v in forward.items()})["f"][0]
        f_minus = build().eval({k: v.copy() for k, v in backward.items()})["f"][0]

        numeric = (f_plus - f_minus) / (2 * step)
        assert np.isclose(analytic["f"][name][0, 0], numeric, rtol=1e-4), \
            f"df/d{name}: analytic {analytic['f'][name][0, 0]}, fd {numeric}"


def test_the_jacobian_does_not_depend_on_what_was_evaluated_before():
    """
    Regression: differentiate() serves its internal eval() from the cache, so
    MDF._eval may not run and the disciplines can still be holding another
    point's converged couplings. The partials were then taken at the wrong
    operating point -- wrong by ~10% on this problem, and silently.
    """
    point_a = {"x1": np.array([1.0]), "z1": np.array([3.0]), "z2": np.array([2.0])}
    point_b = {"x1": np.array([5.0]), "z1": np.array([1.0]), "z2": np.array([8.0])}

    reference = build(CachePolicy.FULL).differentiate(
        {k: v.copy() for k, v in point_a.items()})["f"]["z1"][0, 0]

    prob = build(CachePolicy.FULL)
    prob.eval({k: v.copy() for k, v in point_a.items()})
    prob.eval({k: v.copy() for k, v in point_b.items()})
    after = prob.differentiate(
        {k: v.copy() for k, v in point_a.items()})["f"]["z1"][0, 0]

    assert np.isclose(reference, after, rtol=1e-8), \
        f"reference {reference}, after visiting another point {after}"


def test_evaluation_is_consistent_at_the_same_point():
    """
    The MDA warm-starts from the disciplines' current state, so the same design
    vector must still give the same answer whatever was evaluated in between.
    """
    point_a = {"x1": np.array([1.0]), "z1": np.array([3.0]), "z2": np.array([2.0])}
    point_b = {"x1": np.array([5.0]), "z1": np.array([1.0]), "z2": np.array([8.0])}

    prob = build(CachePolicy.FULL)
    first = prob.eval({k: v.copy() for k, v in point_a.items()})["f"][0]
    prob.eval({k: v.copy() for k, v in point_b.items()})

    fresh = build(CachePolicy.FULL).eval(
        {k: v.copy() for k, v in point_a.items()})["f"][0]

    assert np.isclose(first, fresh, rtol=1e-8)
