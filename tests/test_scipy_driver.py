"""
The SciPy driver, gradient-based and gradient-free.
"""
import numpy as np
import pytest

from msense.api import Constraint, EvaluationFailure, create_driver

from disciplines import Flaky, g, make_problem


def test_slsqp_with_analytic_gradients():
    """
    min x1^2 + x2^2 over the box -> (2, 3), y = 13.
    """
    _, prob = make_problem()
    prob.driver = create_driver(prob, method="SLSQP", n_iter_max=100, tol=1e-9)

    result = prob.solve({"x1": np.array([50.0]), "x2": np.array([50.0])})

    assert result.converged, result.message
    assert np.isclose(result.x["x1"][0], 2.0, atol=1e-4)
    assert np.isclose(result.x["x2"][0], 3.0, atol=1e-4)
    assert np.isclose(result.values["y"][0], 13.0, atol=1e-3)


def test_slsqp_with_an_active_constraint():
    """
    min x1^2 + x2^2 s.t. x1 + x2 >= 10 -> (5, 5), y = 50.
    """
    _, prob = make_problem(constraints=[Constraint(g)])
    prob.driver = create_driver(prob, method="SLSQP", n_iter_max=200, tol=1e-10)

    result = prob.solve({"x1": np.array([50.0]), "x2": np.array([50.0])})

    assert result.converged, result.message
    assert np.isclose(result.x["x1"][0], 5.0, atol=1e-4), result.x
    assert np.isclose(result.x["x2"][0], 5.0, atol=1e-4), result.x
    assert np.isclose(result.values["y"][0], 50.0, atol=1e-3)


def test_cobyla_is_not_handed_a_jacobian():
    _, prob = make_problem(constraints=[Constraint(g)])
    prob.driver = create_driver(prob, method="COBYLA", n_iter_max=500, tol=1e-8)

    assert prob.driver.requires_gradients is False

    result = prob.solve({"x1": np.array([50.0]), "x2": np.array([50.0])})

    assert prob.is_feasible(result.values, tol=1e-4)
    assert result.values["y"][0] < 60.0


def test_history_is_recorded():
    _, prob = make_problem()
    prob.driver = create_driver(prob, method="SLSQP", n_iter_max=100)

    result = prob.solve({"x1": np.array([50.0]), "x2": np.array([50.0])})

    assert len(prob.history) > 1
    assert result.n_iter == len(prob.history)
    # The history holds physical values, and the objective decreases
    assert prob.history[0]["y"][0] > prob.history[-1]["y"][0]
    assert not prob.population_history


def test_result_is_subscriptable_by_variable_name():
    _, prob = make_problem()
    prob.driver = create_driver(prob, method="SLSQP", n_iter_max=50)

    result = prob.solve({"x1": np.array([50.0]), "x2": np.array([50.0])})

    assert result["x1"][0] == result.x["x1"][0]
    assert result["y"][0] == result.values["y"][0]
    with pytest.raises(KeyError):
        result["nope"]


def test_a_failed_evaluation_is_fatal_for_a_gradient_driver():
    disc, prob = make_problem(disc=Flaky())
    prob.driver = create_driver(prob, method="SLSQP", n_iter_max=20)

    with pytest.raises(EvaluationFailure):
        prob.solve({"x1": np.array([80.0]), "x2": np.array([5.0])})


def test_plotting_does_not_raise():
    _, prob = make_problem(constraints=[Constraint(g)])
    prob.driver = create_driver(prob, method="SLSQP", n_iter_max=50)
    prob.solve({"x1": np.array([50.0]), "x2": np.array([50.0])})

    prob.plot_objective_history(show=False)
    prob.plot_variable_history(g, show_lb=True, show=False)
