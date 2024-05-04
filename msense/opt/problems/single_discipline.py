from msense.core.discipline import Discipline
from msense.opt.problems.opt_problem import OptProblem


class SingleDiscipline(OptProblem):
    """
    Single-discipline optimization problem.
    """

    def __init__(self, discipline: Discipline, **kwargs) -> None:
        self.disc = discipline
        super().__init__(**kwargs)

    def _eval(self) -> None:
        self._values.update(self.disc.eval(self._values))

    def _differentiate(self) -> None:
        self._jac.update(self.disc.differentiate(self._values))
