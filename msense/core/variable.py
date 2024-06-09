from dataclasses import dataclass, field

from numpy import ndarray, inf, ones
from numpy import isinf, isneginf

from msense.core.constants import FLOAT_DTYPE


@dataclass(frozen=True)
class Variable:
    name: str = field(default=None, hash=True)
    size: int = field(default=1, hash=False)
    lb: float = field(default=-inf, hash=False)
    ub: float = field(default=inf, hash=False)
    keep_feasible: bool = field(default=False, hash=False)

    def get_bounds_as_array(self, use_normalization: bool = False):
        lb = self.lb * ones(self.size, FLOAT_DTYPE)
        ub = self.ub * ones(self.size, FLOAT_DTYPE)
        keep_feasible = self.keep_feasible * ones(self.size, bool)

        if use_normalization:
            if isinf(self.ub) or isneginf(self.lb) or self.ub == self.lb:
                pass
            else:
                lb, ub = self.norm_values(lb), self.norm_values(ub)

        return lb, ub, keep_feasible

    def norm_values(self, _val: ndarray) -> ndarray:
        return (_val - self.lb) / (self.ub - self.lb)

    def norm_grad(self, _grad: ndarray) -> ndarray:
        return _grad * (self.ub - self.lb)

    def denorm_values(self, _val: ndarray) -> ndarray:
        return _val * (self.ub - self.lb) + self.lb

    def denorm_grad(self, _grad: ndarray) -> ndarray:
        return _grad / (self.ub - self.lb)
