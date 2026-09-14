"""
The formulation surface OptProblem presents to every driver, and the checks it
runs before handing a problem over.
"""
import numpy as np
import pytest

from msense.api import Constraint, DriverCapabilityError, Objective
from msense.api import Variable, create_driver

from disciplines import make_problem, g, y


#
# Dimensions
#

def test_dimensions():
    _, prob = make_problem(objectives=[Objective(y)],
                           constraints=[Constraint(g)])
    assert (prob.n_design, prob.n_obj, prob.n_ineq, prob.n_eq) == (2, 1, 1, 0)


def test_dimensions_with_two_objectives():
    _, prob = make_problem(objectives=[Objective(y), Objective(g, maximize=True)])
    assert prob.n_obj == 2


#
# Objectives
#

def test_objective_array_applies_the_sense():
    values = {"y": np.array([4.0]), "g": np.array([1.0])}
    _, prob_min = make_problem(objectives=[Objective(y)])
    _, prob_max = make_problem(objectives=[Objective(y, maximize=True)])

    assert prob_min.objective_array(values)[0] == 4.0
    assert prob_max.objective_array(values)[0] == -4.0


def test_eval_reports_the_physical_value_not_the_flipped_one():
    _, prob = make_problem(objectives=[Objective(y, maximize=True)])
    out = prob.eval({"x1": np.array([2.0]), "x2": np.array([3.0])})
    assert out["y"][0] == 2.0**2 + 3.0**2


def test_failed_objective_is_large_and_finite():
    """
    A population algorithm computes ideal/nadir points over F, so a failed
    design must not be inf.
    """
    _, prob = make_problem()
    F = prob.objective_array({"y": np.array([np.nan]), "g": np.array([1.0])})
    assert np.isfinite(F[0]) and F[0] == prob.failed_objective_value


def test_scalarization_uses_the_weights():
    _, prob = make_problem(objectives=[Objective(y, weight=2.0),
                                       Objective(g, weight=3.0)],
                           scalarize=True)
    values = {"y": np.array([4.0]), "g": np.array([5.0])}
    assert prob.scalar_objective(values) == 2.0 * 4.0 + 3.0 * 5.0


def test_objective_jacobian_applies_the_sense():
    disc, prob = make_problem(objectives=[Objective(y, maximize=True)])
    jac = prob.differentiate({"x1": np.array([50.0]), "x2": np.array([50.0])})
    J = prob.objective_jac_array(jac)
    # y grows with both design variables, so a maximized y has negative gradient
    assert (J < 0).all()


#
# Constraints
#

def test_constraint_bounds_are_never_normalized():
    """
    use_norm scales design variables. Applying it to a constraint bound would
    map a two-sided bound onto [0, 1] while eval returns the raw value.
    """
    _, prob = make_problem(constraints=[Constraint(g)])
    assert prob.use_norm is True

    cl, cu = prob.constraint_bounds()
    assert cl[0] == 10.0 and np.isinf(cu[0])


def test_two_sided_constraint_bounds_survive_normalization():
    two_sided = Variable("g", lb=10.0, ub=20.0)
    _, prob = make_problem(constraints=[Constraint(two_sided)])
    cl, cu = prob.constraint_bounds()
    assert (cl[0], cu[0]) == (10.0, 20.0)


def test_constraint_row_counts():
    two_sided = Variable("g", lb=10.0, ub=20.0)
    _, prob = make_problem(constraints=[Constraint(two_sided)])
    assert (prob.n_ineq, prob.n_eq) == (2, 0)

    equality = Variable("g", lb=10.0, ub=10.0)
    _, prob = make_problem(constraints=[Constraint(equality)])
    assert (prob.n_ineq, prob.n_eq) == (0, 1)


def test_failed_constraint_is_a_large_violation():
    _, prob = make_problem(constraints=[Constraint(g)])
    gr, _ = prob.constraint_residuals({"y": np.array([1.0]),
                                       "g": np.array([np.nan])})
    assert gr[0] == prob.failed_constraint_violation > 0


def test_is_feasible():
    _, prob = make_problem(constraints=[Constraint(g)])
    assert prob.is_feasible({"y": np.array([1.0]), "g": np.array([11.0])})
    assert not prob.is_feasible({"y": np.array([1.0]), "g": np.array([9.0])})


#
# Evaluation is in physical units; normalization is the driver's business
#

def test_the_problem_evaluates_in_physical_units():
    """
    eval() takes physical values. Normalization exists for the optimizer, and is
    applied at the driver boundary, so the problem, its cache and its history are
    all in the units the discipline was written in.
    """
    _, prob = make_problem()
    assert prob.use_norm is True
    assert prob.eval({"x1": np.array([4.0]), "x2": np.array([5.0])})["y"][0] == 41.0


def test_the_problem_overrides_no_public_evaluation_method():
    """
    Unit conversion inside eval() is what previously made self.eval() unsafe for
    internal callers, which is how a second, private evaluation entry point and
    three different eval_batch overrides came about. Keep it out.
    """
    from msense.api import Discipline, OptProblem
    from msense.opt.problems.single_discipline import SingleDiscipline

    for name in ("eval", "eval_batch", "differentiate"):
        for cls in (OptProblem, SingleDiscipline):
            assert name not in vars(cls), (
                f"{cls.__name__} overrides the public {name}(); "
                f"override the _{name} hook instead")
        assert name in vars(Discipline)


def test_the_driver_boundary_converts_both_ways():
    from msense.api import create_driver

    _, prob = make_problem()
    driver = create_driver(prob)

    x = driver.design_array({"x1": np.array([2.0]), "x2": np.array([90.0])})
    assert np.allclose(x, [0.0, 1.0])           # normalized for the optimizer

    physical = driver.design_dict(np.array([0.0, 1.0]))
    assert physical["x1"][0] == 2.0 and physical["x2"][0] == 90.0


#
# Design bounds and normalization
#

def test_design_bounds_normalized_and_physical():
    _, prob = make_problem()
    xl, xu = prob.design_bounds(use_norm=True)
    assert np.allclose(xl, [0.0, 0.0]) and np.allclose(xu, [1.0, 1.0])

    xl, xu = prob.design_bounds(use_norm=False)
    assert np.allclose(xl, [2.0, 3.0]) and np.allclose(xu, [100.0, 90.0])


def test_normalization_is_disabled_by_an_unbounded_design_variable():
    unbounded = Variable("x1", lb=2.0)
    _, prob = make_problem()
    assert prob.use_norm is True

    from msense.api import create_opt_problem
    from disciplines import Parabola, x2
    prob = create_opt_problem("single_discipline", [Parabola()],
                              design_vars=[unbounded, x2],
                              objectives=[Objective(y)], name="Unbounded")
    assert prob.use_norm is False


def test_denormalize_round_trip():
    _, prob = make_problem()
    physical = prob.denormalize({"x1": np.array([0.0]), "x2": np.array([1.0])})
    assert physical["x1"][0] == 2.0 and physical["x2"][0] == 90.0


#
# Capability checks
#

def test_multi_objective_on_a_single_objective_driver_is_refused():
    _, prob = make_problem(objectives=[Objective(y), Objective(g, maximize=True)])
    prob.driver = create_driver(prob, method="SLSQP")

    with pytest.raises(DriverCapabilityError, match="objectives"):
        prob.solve({"x1": np.array([50.0]), "x2": np.array([50.0])})


def test_multi_objective_scalarization_when_requested():
    _, prob = make_problem(objectives=[Objective(y), Objective(g)],
                           scalarize=True)
    prob.driver = create_driver(prob, method="SLSQP", n_iter_max=50)

    result = prob.solve({"x1": np.array([50.0]), "x2": np.array([50.0])})

    assert result.converged, result.message
    assert prob.n_obj == 2 and not result.has_front


def test_at_least_one_objective_is_required():
    from msense.api import create_opt_problem
    from disciplines import Parabola, x1, x2

    with pytest.raises(ValueError, match="objective"):
        create_opt_problem("single_discipline", [Parabola()],
                           design_vars=[x1, x2], objectives=[], name="NoObj")


def test_deferred_formulations_are_reported_clearly():
    from msense.api import create_opt_problem

    with pytest.raises(NotImplementedError, match="stateless"):
        create_opt_problem("mdf", [], [], objectives=[])
