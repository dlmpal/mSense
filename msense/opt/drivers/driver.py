from typing import Dict

from numpy import ndarray

from msense.core.discipline import Discipline


class Driver:
    """
    Base driver class
    """

    def __init__(self, disc: Discipline):
        self.disc = disc
        self.callback = None
        self.iter = 0

    def solve(self, input_values: Dict[str, ndarray], use_norm: bool):
        raise NotImplementedError
