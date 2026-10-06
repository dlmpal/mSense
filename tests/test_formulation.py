"""
The canonical reading of objectives and constraints.
"""
import numpy as np
import pytest

from msense.api import Constraint, ConstraintKind, Objective, Variable
from msense.opt.formulation import as_constraints, as_objectives


#
# Objectives
#

def test_objective_sense():
    assert Objective(Variable("y")).sense == 1.0
    assert Objective(Variable("y"), maximize=True).sense == -1.0


def test_bare_variable_becomes_a_minimized_objective():
    obj = as_objectives(Variable("y"))
    assert len(obj) == 1 and obj[0].maximize is False


def test_bare_variable_becomes_a_constraint():
    con = as_constraints(Variable("c", lb=1.0))
    assert len(con) == 1 and isinstance(con[0], Constraint)


#
# Constraint kind and row count
#

def test_lower_bound_only():
    con = Constraint(Variable("c", lb=1.0))
    assert con.kind == ConstraintKind.INEQUALITY and con.n_rows == 1


def test_upper_bound_only():
    con = Constraint(Variable("c", ub=1.0))
    assert con.kind == ConstraintKind.INEQUALITY and con.n_rows == 1


def test_two_sided_contributes_both_bounds():
    con = Constraint(Variable("c", lb=-1.0, ub=1.0))
    assert con.kind == ConstraintKind.INEQUALITY and con.n_rows == 2


def test_equality():
    con = Constraint(Variable("c", lb=1.0, ub=1.0))
    assert con.kind == ConstraintKind.EQUALITY
    assert con.n_rows == 1 and con.is_equality


def test_a_constraint_with_no_finite_bound_is_rejected():
    """
    It constrains nothing, and allowing it means filtering it out at every
    point of use.
    """
    with pytest.raises(ValueError, match="no finite bound"):
        Constraint(Variable("c"))


def test_row_counts_scale_with_variable_size():
    assert Constraint(Variable("c", size=3, lb=-1.0, ub=1.0)).n_rows == 6
    assert Constraint(Variable("c", size=3, lb=1.0)).n_rows == 3
    assert Constraint(Variable("c", size=3, lb=1.0, ub=1.0)).n_rows == 3


#
# Residual sign convention: satisfied <= 0, violated > 0
#

def test_lower_bound_residual_sign():
    con = Constraint(Variable("c", lb=1.0))
    assert con.residuals(np.array([2.0]))[0] < 0     # satisfied
    assert con.residuals(np.array([0.0]))[0] > 0     # violated
    assert con.residuals(np.array([1.0]))[0] == 0    # active


def test_upper_bound_residual_sign():
    con = Constraint(Variable("c", ub=1.0))
    assert con.residuals(np.array([0.0]))[0] < 0     # satisfied
    assert con.residuals(np.array([2.0]))[0] > 0     # violated


def test_two_sided_residuals():
    con = Constraint(Variable("c", lb=-1.0, ub=2.0))
    assert (con.residuals(np.array([0.0])) < 0).all()      # inside both bounds

    rows = con.residuals(np.array([3.0]))                  # above the upper bound
    assert rows[0] < 0 and rows[1] > 0


def test_equality_residual():
    con = Constraint(Variable("c", lb=1.0, ub=1.0))
    assert con.residuals(np.array([1.5]))[0] == 0.5
    assert con.residuals(np.array([1.0]))[0] == 0.0


#
# Normalized residuals: each row divided by the magnitude of its bound
#

def test_normalized_residuals_are_relative_to_the_bound():
    con = Constraint(Variable("c", lb=10.0))
    assert np.allclose(con.residuals(np.array([9.0]), normalized=True), [0.1])
    assert np.allclose(con.residuals(np.array([9.0])), [1.0])


def test_normalized_residuals_keep_their_sign_for_a_negative_bound():
    # A lower coil's zone edge: Z_max <= -0.5
    con = Constraint(Variable("z", ub=-0.5))
    assert con.residuals(np.array([-0.6]), normalized=True)[0] < 0      # inside: satisfied
    assert np.allclose(con.residuals(np.array([-0.4]), normalized=True), [0.2])
    con = Constraint(Variable("z", lb=-0.5))
    assert np.allclose(con.residuals(np.array([-0.6]), normalized=True), [0.2])


def test_a_zero_bound_is_left_unscaled():
    con = Constraint(Variable("c", lb=0.0, ub=4.0))
    assert np.allclose(con.residuals(np.array([-1.0]), normalized=True), [1.0, -1.25])


def test_constraint_scale():
    assert Constraint(Variable("c", lb=-20.0, ub=5.0)).scale() == 20.0
    assert Constraint(Variable("c", lb=0.0)).scale() == 1.0
    assert Constraint(Variable("c", ub=-3.0)).scale() == 3.0
    assert Constraint(Variable("c", ub=-3.0)).scale(normalized=False) == 1.0
