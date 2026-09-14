"""
The MDA solvers: nonlinear Gauss-Seidel, nonlinear Jacobi and Newton-Raphson.

These drive a coupled system to its fixed point. Both test systems have an
analytic solution, so the assertions are against a known answer rather than
against the solvers' own output.
"""
import numpy as np
import pytest

from msense.api import Solver, SolverType, create_solver
from msense.api import NonlinearGS, NonlinearJacobi, NewtonRaphson

from disciplines import LinearCoupling1, LinearCoupling2, linear_coupling_solution
from disciplines import SellarCoupling1, SellarCoupling2, sellar_coupling_solution

ALL_TYPES = [SolverType.NONLINEAR_GS,
             SolverType.NONLINEAR_JACOBI,
             SolverType.NEWTON_RAPHSON]

START = {"y1": np.array([1.0]), "y2": np.array([1.0])}


def linear_solver(solver_type, **kwargs):
    kwargs.setdefault("n_iter_max", 50)
    kwargs.setdefault("tol", 1e-10)
    return create_solver([LinearCoupling1(), LinearCoupling2()],
                         solver_type, **kwargs)


def sellar_solver(solver_type, **kwargs):
    kwargs.setdefault("n_iter_max", 50)
    kwargs.setdefault("tol", 1e-10)
    return create_solver([SellarCoupling1(), SellarCoupling2()],
                         solver_type, **kwargs)


#
# The factory
#

def test_factory_returns_the_requested_solver():
    disciplines = [LinearCoupling1(), LinearCoupling2()]
    assert isinstance(create_solver(disciplines, SolverType.NONLINEAR_GS), NonlinearGS)
    assert isinstance(create_solver(disciplines, SolverType.NONLINEAR_JACOBI), NonlinearJacobi)
    assert isinstance(create_solver(disciplines, SolverType.NEWTON_RAPHSON), NewtonRaphson)


def test_factory_defaults_to_gauss_seidel():
    assert isinstance(create_solver([LinearCoupling1(), LinearCoupling2()]),
                      NonlinearGS)


def test_couplings_are_identified():
    """
    The solver has to work out for itself which variables couple the
    disciplines: y1 and y2 are each an output of one and an input of the other.
    """
    solver = linear_solver(SolverType.NONLINEAR_GS)
    assert sorted(var.name for var in solver.coupling_vars) == ["y1", "y2"]


#
# Every solver reaches the known fixed point
#

@pytest.mark.parametrize("solver_type", ALL_TYPES)
def test_linear_system_is_solved(solver_type):
    y1_exact, y2_exact = linear_coupling_solution()

    solver = linear_solver(solver_type)
    values = solver.solve(dict(START))

    assert solver.status == Solver.SolverStatus.CONVERGED
    assert np.isclose(values["y1"][0], y1_exact, atol=1e-8), values
    assert np.isclose(values["y2"][0], y2_exact, atol=1e-8), values


@pytest.mark.parametrize("solver_type", ALL_TYPES)
def test_nonlinear_system_is_solved(solver_type):
    y1_exact, y2_exact = sellar_coupling_solution()

    solver = sellar_solver(solver_type)
    values = solver.solve(dict(START))

    assert solver.status == Solver.SolverStatus.CONVERGED
    assert np.isclose(values["y1"][0], y1_exact, atol=1e-8), values
    assert np.isclose(values["y2"][0], y2_exact, atol=1e-8), values


#
# Properties each method is supposed to have
#

def test_newton_solves_a_linear_system_in_one_iteration():
    """
    Newton-Raphson is exact for a linear system, so the first correction lands
    on the solution and the second iteration only confirms it.
    """
    y1_exact, _ = linear_coupling_solution()

    solver = linear_solver(SolverType.NEWTON_RAPHSON)
    values = solver.solve(dict(START))

    assert solver.iter <= 2, f"took {solver.iter} iterations"
    assert np.isclose(values["y1"][0], y1_exact, atol=1e-12)


def test_gauss_seidel_is_faster_than_jacobi():
    """
    Gauss-Seidel feeds each discipline the values already updated in this
    sweep, so it should need fewer iterations than Jacobi, which uses only the
    previous sweep.
    """
    gs = sellar_solver(SolverType.NONLINEAR_GS)
    jacobi = sellar_solver(SolverType.NONLINEAR_JACOBI)

    gs.solve(dict(START))
    jacobi.solve(dict(START))

    assert gs.status == jacobi.status == Solver.SolverStatus.CONVERGED
    assert gs.iter < jacobi.iter, f"GS {gs.iter}, Jacobi {jacobi.iter}"


@pytest.mark.parametrize("solver_type", ALL_TYPES)
def test_the_residual_history_decreases(solver_type):
    solver = sellar_solver(solver_type)
    solver.solve(dict(START))

    assert len(solver.history) > 1
    assert solver.history[-1] < solver.history[0]
    assert solver.history[-1] <= solver.tol


@pytest.mark.parametrize("solver_type", ALL_TYPES)
def test_starting_from_the_solution_is_a_fixed_point(solver_type):
    """
    Started at the answer, a solver must stay there.
    """
    y1_exact, y2_exact = sellar_coupling_solution()
    start = {"y1": np.array([y1_exact]), "y2": np.array([y2_exact])}

    solver = sellar_solver(solver_type)
    values = solver.solve(start)

    assert np.isclose(values["y1"][0], y1_exact, atol=1e-10)
    assert np.isclose(values["y2"][0], y2_exact, atol=1e-10)


#
# Convergence control
#

def test_not_converging_is_reported_not_hidden():
    solver = sellar_solver(SolverType.NONLINEAR_JACOBI, n_iter_max=2, tol=1e-12)
    solver.solve(dict(START))

    assert solver.status == Solver.SolverStatus.NOT_CONVERGED
    assert solver.iter == 2


def test_under_relaxation_still_converges():
    """
    Relaxation trades iterations for stability; it must not change the answer.
    """
    y1_exact, _ = sellar_coupling_solution()

    relaxed = sellar_solver(SolverType.NONLINEAR_GS, relax_fact=0.5)
    values = relaxed.solve(dict(START))

    assert relaxed.status == Solver.SolverStatus.CONVERGED
    assert np.isclose(values["y1"][0], y1_exact, atol=1e-8)


def test_a_looser_tolerance_stops_sooner():
    loose = sellar_solver(SolverType.NONLINEAR_GS, tol=1e-3)
    tight = sellar_solver(SolverType.NONLINEAR_GS, tol=1e-12)

    loose.solve(dict(START))
    tight.solve(dict(START))

    assert loose.iter < tight.iter


#
# Interaction with the stateless core
#

def test_the_solver_does_not_mutate_the_caller_s_values():
    solver = sellar_solver(SolverType.NONLINEAR_GS)
    start = dict(START)
    before = {k: v.copy() for k, v in start.items()}

    solver.solve(start)

    for name in before:
        assert np.array_equal(start[name], before[name]), name


def test_solving_twice_gives_the_same_answer():
    """
    Nothing may carry over between solves.
    """
    solver = sellar_solver(SolverType.NONLINEAR_GS)

    first = solver.solve(dict(START))
    iterations = solver.iter
    second = solver.solve(dict(START))

    assert np.isclose(first["y1"][0], second["y1"][0], atol=1e-12)
    assert solver.iter == iterations


def test_disciplines_are_reusable_across_solvers():
    """
    The same discipline objects handed to two solvers must not interfere --
    evaluation is stateless, so this has to hold.
    """
    disciplines = [SellarCoupling1(), SellarCoupling2()]
    y1_exact, _ = sellar_coupling_solution()

    gs = create_solver(disciplines, SolverType.NONLINEAR_GS, n_iter_max=50, tol=1e-10)
    nr = create_solver(disciplines, SolverType.NEWTON_RAPHSON, n_iter_max=50, tol=1e-10)

    from_gs = gs.solve(dict(START))
    from_nr = nr.solve(dict(START))

    assert np.isclose(from_gs["y1"][0], y1_exact, atol=1e-8)
    assert np.isclose(from_nr["y1"][0], y1_exact, atol=1e-8)
