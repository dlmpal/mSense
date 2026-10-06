"""
The NEOS driver: metamodel-assisted search, Pareto fronts, failure tolerance.
"""
import numpy as np
import pytest

from msense.api import Constraint, Objective, Variable, create_driver
from msense.opt.drivers.neos_driver import MSENSE_HAS_NEOS

from disciplines import Flaky, g, make_problem, y

pytestmark = pytest.mark.skipif(not MSENSE_HAS_NEOS,
                                reason="neos is not installed")

if MSENSE_HAS_NEOS:
    from neos import LocalRBF


def small_surrogate():
    # Starts pre-evaluating after two generations of the small populations below
    return LocalRBF(min_db=40, min_db_ok=20, n_patterns=(8, 10))


def neos_driver(prob, **kwargs):
    kwargs.setdefault("pop_size", 10)
    kwargs.setdefault("n_offsprings", 20)
    kwargs.setdefault("surrogate", small_surrogate())
    return create_driver(prob, type="neos_driver", **kwargs)


#
# Single objective
#

def test_approaches_the_optimum_within_the_budget():
    disc, prob = make_problem()
    prob.driver = neos_driver(prob, n_eval_max=150, seed=1)

    result = prob.solve({"x1": np.array([50.0]), "x2": np.array([50.0])})

    assert result.converged, result.message
    assert np.isclose(result.values["y"][0], 13.0, rtol=0.05), result.values["y"]
    assert result.n_eval == disc.n_eval <= 150


def test_the_surrogate_saves_evaluations():
    _, prob = make_problem()
    prob.driver = neos_driver(prob, n_iter_max=10, seed=1)

    result = prob.solve()

    # Two full generations, then a few exact evaluations per generation
    assert result.n_eval < 10 * 20 / 2


def test_respects_an_inequality_constraint():
    # g = x1 + x2 >= 10 is active at the optimum (5, 5)
    _, prob = make_problem(constraints=[Constraint(g)])
    # The penalty is added to y (~50 here), so it has to be steep to matter.
    # The residual is relative to the bound: a max violation of 0.1 is g < 9.
    prob.driver = neos_driver(prob, n_eval_max=200, seed=1, max_violation=0.1,
                              constraint_slope=100.0)

    result = prob.solve()

    assert result.converged, result.message
    assert 10.0 <= result.values["g"][0] < 10.1, result.values["g"]


def test_raw_residuals_without_normalization():
    # Without normalization the residual is in g's own units: a max violation of 1 is g < 9
    _, prob = make_problem(constraints=[Constraint(g)], use_norm=False)
    prob.driver = neos_driver(prob, n_eval_max=200, seed=1,
                              max_violation=1.0, constraint_slope=10.0)

    result = prob.solve()

    assert result.converged, result.message
    assert 10.0 <= result.values["g"][0] < 10.1, result.values["g"]


def test_penalty_settings_map_onto_neos():
    _, prob = make_problem(constraints=[Constraint(g)])
    driver = neos_driver(prob, constraint_slope=30.0, max_violation=2.0)
    assert driver.options["constraint_fail"] == 2.0
    assert driver.options["constraint_intensity"] == 60.0


def test_raw_neos_penalty_keywords_are_refused():
    _, prob = make_problem(constraints=[Constraint(g)])
    with pytest.raises(ValueError, match="constraint_slope"):
        neos_driver(prob, constraint_fail=100.0)


def test_surrogate_from_a_dictionary():
    _, prob = make_problem()
    driver = create_driver(prob, type="neos_driver", n_iter_max=2,
                           surrogate={"n_exact": [2, 3], "min_db": 40})

    assert isinstance(driver.options["surrogate"], LocalRBF)
    assert driver.options["surrogate"].n_exact == (2, 3)
    assert driver.options["surrogate"].min_db == 40


def test_equality_constraints_are_refused():
    equal = Variable("g", lb=10.0, ub=10.0)
    _, prob = make_problem(constraints=[Constraint(equal)])
    prob.driver = neos_driver(prob, n_iter_max=2)

    with pytest.raises(Exception, match="equality"):
        prob.solve()


def test_one_history_entry_per_generation():
    _, prob = make_problem()
    prob.driver = neos_driver(prob, n_iter_max=6, seed=1)

    result = prob.solve()

    assert result.n_iter == len(prob.population_history) == len(prob.history) == 6
    assert all(len(generation) > 0 for generation in prob.population_history)


def test_starting_point_seeds_generation_zero():
    _, prob = make_problem()
    prob.driver = neos_driver(prob, n_iter_max=1, seed=3)

    prob.solve({"x1": np.array([7.0]), "x2": np.array([11.0])})

    assert any(np.isclose(m["x1"][0], 7.0) and np.isclose(m["x2"][0], 11.0)
               for m in prob.population_history[0])


def test_the_same_seed_reproduces_the_run():
    def run(seed):
        _, prob = make_problem()
        prob.driver = neos_driver(prob, n_iter_max=4, seed=seed)
        prob.solve()
        return np.array([[m["x1"][0], m["x2"][0]]
                         for m in prob.population_history[-1]])

    assert np.array_equal(run(1), run(1))
    assert not np.array_equal(run(1), run(2))


def test_works_without_normalization():
    _, prob = make_problem(use_norm=False)
    prob.driver = neos_driver(prob, n_eval_max=150, seed=1)

    result = prob.solve()

    assert 2.0 <= result.x["x1"][0] <= 100.0 and 3.0 <= result.x["x2"][0] <= 90.0
    assert np.isclose(result.values["y"][0], 13.0, rtol=0.05)


#
# Multiple objectives
#

def test_returns_a_front_spanning_a_real_trade_off():
    _, prob = make_problem(objectives=[Objective(y), Objective(g, maximize=True)])
    prob.driver = neos_driver(prob, n_eval_max=300, seed=1, archive_size=20)

    result = prob.solve()

    assert result.has_front and 3 < len(result.front_values) <= 20
    ys = [v["y"][0] for v in result.front_values]
    gs = [v["g"][0] for v in result.front_values]
    assert max(ys) - min(ys) > 1.0 and max(gs) - min(gs) > 1.0
    assert np.corrcoef(ys, gs)[0, 1] > 0.9
    for x in result.front_x:
        assert 2.0 <= x["x1"][0] <= 100.0 and 3.0 <= x["x2"][0] <= 90.0


#
# Failure tolerance
#

def test_failed_evaluations_are_absorbed():
    disc, prob = make_problem(disc=Flaky())
    prob.driver = neos_driver(prob, n_eval_max=150, seed=1)

    result = prob.solve()

    assert disc.n_fail > 0
    assert result.converged, result.message
    assert result.x["x1"][0] <= 50.0
    assert np.isclose(result.values["y"][0], 13.0, rtol=0.05)
