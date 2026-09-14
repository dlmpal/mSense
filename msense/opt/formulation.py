from typing import List, Tuple
from dataclasses import dataclass, field
from enum import Enum

from numpy import ndarray, isinf, isneginf, concatenate

from msense.core.constants import FLOAT_DTYPE
from msense.core.variable import Variable


@dataclass(frozen=True)
class Objective:
    """
    An objective: a variable, plus the sense in which it is to be optimized.

    Args:
        var (Variable): The variable holding the objective value.
        maximize (bool, optional): Whether the objective is to be maximized. Defaults to False.
        weight (float, optional): Relative weight, used only when a driver that cannot
            handle multiple objectives is asked to scalarize them. Defaults to 1.0.
    """
    var: Variable = field(hash=True)
    maximize: bool = field(default=False, hash=False)
    weight: float = field(default=1.0, hash=False)

    @property
    def name(self) -> str:
        return self.var.name

    @property
    def sense(self) -> float:
        """
        The factor converting the objective to minimization form.
        """
        return -1.0 if self.maximize else 1.0


class ConstraintKind(str, Enum):
    """
    Whether a constraint is satisfied at zero or below it.
    """
    INEQUALITY = "inequality"    # residual <= 0
    EQUALITY = "equality"        # residual == 0


@dataclass(frozen=True)
class Constraint:
    """
    A constraint: a variable whose lower and/or upper bound must be respected.

    The bounds live on the Variable, as they always have. What this class adds is
    a single canonical reading of them, so that every driver consumes the same
    convention instead of re-deriving one:

    * lb only          -> inequality, one row,   lb - c <= 0
    * ub only          -> inequality, one row,   c - ub <= 0
    * lb and ub, both finite and different -> inequality, both rows
    * lb == ub, finite -> equality,   one row,   c - lb == 0

    A constraint with no finite bound constrains nothing and is rejected at
    construction: it would otherwise have to be filtered out at every point of
    use.
    """
    var: Variable = field(hash=True)

    def __post_init__(self) -> None:
        if not self.has_lb and not self.has_ub:
            raise ValueError(
                f"Constraint on variable '{self.var.name}' has no finite bound, "
                f"so it constrains nothing. Give the Variable an lb, a ub, or "
                f"both, or leave it out of the constraint list.")

    @property
    def name(self) -> str:
        return self.var.name

    @property
    def has_lb(self) -> bool:
        return not isinf(self.var.lb) and not isneginf(self.var.lb)

    @property
    def has_ub(self) -> bool:
        return not isinf(self.var.ub) and not isneginf(self.var.ub)

    @property
    def kind(self) -> ConstraintKind:
        """
        Whether this constraint is an equality or an inequality.
        """
        if self.has_lb and self.has_ub and self.var.lb == self.var.ub:
            return ConstraintKind.EQUALITY
        return ConstraintKind.INEQUALITY

    @property
    def is_equality(self) -> bool:
        return self.kind == ConstraintKind.EQUALITY

    @property
    def n_rows(self) -> int:
        """
        The number of scalar residual rows this constraint contributes.

        A two-sided inequality contributes both of its bounds.
        """
        if self.is_equality:
            return self.var.size
        return self.var.size * ((1 if self.has_lb else 0) +
                                (1 if self.has_ub else 0))

    def residuals(self, value: ndarray) -> ndarray:
        """
        Convert a constraint value into residual rows.

        Args:
            value (ndarray): The constraint value, of size var.size.

        Returns:
            ndarray: The residuals, of size n_rows. An equality is satisfied
                where they are zero, an inequality where they are non-positive.
        """
        if self.is_equality:
            return value - self.var.lb

        rows = []
        if self.has_lb:
            rows.append(self.var.lb - value)
        if self.has_ub:
            rows.append(value - self.var.ub)

        return concatenate(rows)


def as_objectives(objectives) -> List[Objective]:
    """
    Accept a Variable, an Objective, or a list of either, and return Objectives.
    """
    if objectives is None:
        return []
    if isinstance(objectives, (Variable, Objective)):
        objectives = [objectives]
    return [obj if isinstance(obj, Objective) else Objective(obj) for obj in objectives]


def as_constraints(constraints) -> List[Constraint]:
    """
    Accept a Variable, a Constraint, or a list of either, and return Constraints.
    """
    if constraints is None:
        return []
    if isinstance(constraints, (Variable, Constraint)):
        constraints = [constraints]
    return [con if isinstance(con, Constraint) else Constraint(con) for con in constraints]
