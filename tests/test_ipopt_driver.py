"""
The Ipopt driver. Skipped where cyipopt is unavailable.
"""
import numpy as np
import pytest

from msense.api import Constraint, create_driver
from msense.opt.drivers.ipopt_driver import MSENSE_HAS_IPOPT

from disciplines import g, make_problem

pytestmark = pytest.mark.skipif(not MSENSE_HAS_IPOPT,
                                reason="cyipopt is not installed")


def test_ipopt_with_an_active_constraint():
    """
    min x1^2 + x2^2 s.t. x1 + x2 >= 10 -> (5, 5), y = 50.
    """
    _, prob = make_problem(constraints=[Constraint(g)])
    prob.driver = create_driver(prob, type="ipopt_driver",
                                n_iter_max=200, tol=1e-9)

    result = prob.solve({"x1": np.array([50.0]), "x2": np.array([50.0])})

    assert result.converged, result.message
    assert np.isclose(result.x["x1"][0], 5.0, atol=1e-4), result.x
    assert np.isclose(result.x["x2"][0], 5.0, atol=1e-4), result.x
    assert np.isclose(result.values["y"][0], 50.0, atol=1e-3)


def test_ipopt_reports_success_as_converged():
    """
    Ipopt's success status is 0, which is falsy; reporting it directly inverted
    the convergence flag.
    """
    _, prob = make_problem()
    prob.driver = create_driver(prob, type="ipopt_driver", n_iter_max=200)

    result = prob.solve({"x1": np.array([50.0]), "x2": np.array([50.0])})

    assert result.converged is True, result.message
    assert np.isclose(result.values["y"][0], 13.0, atol=1e-3)
