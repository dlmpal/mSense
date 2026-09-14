"""
The pymoo driver: population-based search, Pareto fronts, failure tolerance.
"""
import numpy as np
import pytest

from msense.api import Constraint, Objective, Variable, create_driver
from msense.opt.drivers.pymoo_driver import MSENSE_HAS_PYMOO

from disciplines import Flaky, g, make_problem, y

pytestmark = pytest.mark.skipif(not MSENSE_HAS_PYMOO,
                                reason="pymoo is not installed")


#
# Single objective
#

def test_ga_approaches_the_optimum():
    _, prob = make_problem()
    prob.driver = create_driver(prob, type="pymoo_driver", algorithm="GA",
                                pop_size=20, n_iter_max=25, seed=1)

    result = prob.solve({"x1": np.array([50.0]), "x2": np.array([50.0])})

    assert result.converged, result.message
    assert np.isclose(result.values["y"][0], 13.0, rtol=0.05), result.values["y"]


def test_one_history_entry_per_generation():
    _, prob = make_problem()
    prob.driver = create_driver(prob, type="pymoo_driver", algorithm="GA",
                                pop_size=10, n_iter_max=12, seed=1)

    result = prob.solve({"x1": np.array([50.0]), "x2": np.array([50.0])})

    assert len(prob.population_history) == 12
    assert len(prob.history) == 12
    assert all(len(gen) == 10 for gen in prob.population_history)
    assert result.n_iter == 12


def test_starting_point_seeds_generation_zero():
    """
    A converged baseline is expensive information; it should not be discarded.
    """
    _, prob = make_problem()
    prob.driver = create_driver(prob, type="pymoo_driver", algorithm="GA",
                                pop_size=10, n_iter_max=1, seed=3)

    prob.solve({"x1": np.array([7.0]), "x2": np.array([11.0])})

    gen0 = prob.population_history[0]
    assert any(np.isclose(m["x1"][0], 7.0) and np.isclose(m["x2"][0], 11.0)
               for m in gen0), "starting point was not carried into generation 0"


def test_the_same_seed_reproduces_the_run():
    """
    The initial population is drawn before pymoo's minimize() applies the seed,
    so the driver has to seed it itself or the run is not reproducible.
    """
    def generation_zero(seed):
        _, prob = make_problem()
        prob.driver = create_driver(prob, type="pymoo_driver", algorithm="GA",
                                    pop_size=8, n_iter_max=2, seed=seed)
        prob.solve({"x1": np.array([50.0]), "x2": np.array([50.0])})
        return np.array([[m["x1"][0], m["x2"][0]]
                         for m in prob.population_history[0]])

    assert np.allclose(generation_zero(1), generation_zero(1))
    assert not np.allclose(generation_zero(1), generation_zero(2))


def test_starting_point_is_optional():
    _, prob = make_problem()
    prob.driver = create_driver(prob, type="pymoo_driver", algorithm="GA",
                                pop_size=10, n_iter_max=5, seed=1)

    result = prob.solve()

    assert result.converged, result.message


def test_evaluation_count_reflects_the_population():
    disc, prob = make_problem()
    prob.driver = create_driver(prob, type="pymoo_driver", algorithm="GA",
                                pop_size=10, n_iter_max=8, seed=1)

    result = prob.solve()

    # Bounded above by pop_size * n_gen; below by the first generation
    assert 10 <= result.n_eval <= 80
    assert result.n_eval == disc.n_eval


#
# Multiple objectives
#

def test_nsga2_returns_a_front_spanning_a_real_trade_off():
    # Minimize y and maximize g: in tension over the box
    _, prob = make_problem(objectives=[Objective(y), Objective(g, maximize=True)])
    prob.driver = create_driver(prob, type="pymoo_driver", algorithm="NSGA2",
                                pop_size=40, n_iter_max=30, seed=1)

    result = prob.solve({"x1": np.array([50.0]), "x2": np.array([50.0])})

    assert result.has_front and len(result.front_values) > 3
    ys = [v["y"][0] for v in result.front_values]
    gs = [v["g"][0] for v in result.front_values]

    assert max(ys) - min(ys) > 1.0 and max(gs) - min(gs) > 1.0
    # More g must cost more y, or it is not a trade-off
    assert np.corrcoef(ys, gs)[0, 1] > 0.9


def test_nsga2_respects_a_two_sided_constraint():
    two_sided = Variable("g", lb=10.0, ub=20.0)
    _, prob = make_problem(objectives=[Objective(y), Objective(g, maximize=True)],
                           constraints=[Constraint(two_sided)])
    prob.driver = create_driver(prob, type="pymoo_driver", algorithm="NSGA2",
                                pop_size=40, n_iter_max=40, seed=1)

    result = prob.solve({"x1": np.array([5.0]), "x2": np.array([5.0])})

    assert result.has_front
    for values in result.front_values:
        assert 10.0 - 1e-3 <= values["g"][0] <= 20.0 + 1e-3, values["g"]


def test_front_members_are_reported_in_physical_units():
    _, prob = make_problem(objectives=[Objective(y), Objective(g, maximize=True)])
    prob.driver = create_driver(prob, type="pymoo_driver", algorithm="NSGA2",
                                pop_size=20, n_iter_max=10, seed=1)

    result = prob.solve()

    for x in result.front_x:
        assert 2.0 <= x["x1"][0] <= 100.0
        assert 3.0 <= x["x2"][0] <= 90.0


def test_pareto_front_plot_does_not_raise():
    _, prob = make_problem(objectives=[Objective(y), Objective(g, maximize=True)])
    prob.driver = create_driver(prob, type="pymoo_driver", algorithm="NSGA2",
                                pop_size=20, n_iter_max=10, seed=1)
    prob.solve()

    prob.plot_pareto_front(show=False)
    prob.plot_objective_history(show=False)


#
# Algorithms
#

@pytest.mark.parametrize("algorithm", ["GA", "DE", "PSO"])
def test_single_objective_algorithms_run(algorithm):
    _, prob = make_problem()
    prob.driver = create_driver(prob, type="pymoo_driver", algorithm=algorithm,
                                pop_size=20, n_iter_max=10, seed=1)

    result = prob.solve({"x1": np.array([50.0]), "x2": np.array([50.0])})

    assert result.converged, result.message
    assert np.isfinite(result.values["y"][0])


def test_nsga3_runs():
    _, prob = make_problem(objectives=[Objective(y), Objective(g, maximize=True)])
    prob.driver = create_driver(prob, type="pymoo_driver", algorithm="NSGA3",
                                pop_size=20, n_iter_max=10, seed=1,
                                n_partitions=8)

    result = prob.solve()

    assert result.has_front


def test_unknown_algorithm_is_reported():
    _, prob = make_problem()
    prob.driver = create_driver(prob, type="pymoo_driver", algorithm="NOPE",
                                pop_size=10, n_iter_max=2)

    with pytest.raises(ValueError, match="NOPE"):
        prob.solve()


#
# Failure tolerance
#

def test_failed_evaluations_are_absorbed():
    disc, prob = make_problem(disc=Flaky())
    prob.driver = create_driver(prob, type="pymoo_driver", algorithm="GA",
                                pop_size=20, n_iter_max=15, seed=1)

    result = prob.solve()

    assert disc.n_fail > 0, "the flaky discipline should have failed at least once"
    assert result.converged, result.message
    # The search should end up in the region that solves
    assert np.isfinite(result.values["y"][0])
    assert result.values["y"][0] < 50.0**2


def test_the_driver_declares_that_it_tolerates_failure():
    _, prob = make_problem()
    driver = create_driver(prob, type="pymoo_driver")
    assert driver.tolerates_failed_evaluations is True
    assert driver.is_population_based is True
    assert driver.supports_multi_objective is True
