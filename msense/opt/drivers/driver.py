from typing import Callable, Dict, Tuple

from numpy import ndarray

from msense.core.discipline import Discipline


class Driver:
    """
    Base driver class
    """

    def __init__(self, discipline: Discipline, n_iter_max: int = 10, tol: float = 1e-6, callback: Callable = None):
        self.disc = discipline
        self.n_iter_max = n_iter_max
        self.tol = tol
        self.callback = callback
        self.iter = 0

    def _callback(self) -> None:
        if self.callback is not None:
            self.callback()
        self.iter += 1

    def solve(self, input_values: Dict[str, ndarray], use_norm: bool) -> Tuple[bool, str]:
        raise NotImplementedError
