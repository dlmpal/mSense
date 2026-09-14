from typing import Dict, List, Mapping, Sequence

from numpy import ndarray

from msense.core.discipline import BatchOutcome, Discipline
from msense.opt.problems.opt_problem import OptProblem


class SingleDiscipline(OptProblem):
    """
    Single-discipline optimization problem.
    """

    def __init__(self, discipline: Discipline, **kwargs) -> None:
        self.disc = discipline
        super().__init__(**kwargs)

    def _eval(self, inputs: Mapping[str, ndarray]) -> Dict[str, ndarray]:
        return self.disc.eval(inputs)

    def _eval_batch(self, input_rows: Sequence[Mapping[str, ndarray]]) -> List[BatchOutcome]:
        """
        Hand the whole batch to the discipline, so that a discipline able to
        dispatch its evaluations concurrently gets the chance to.
        """
        return self.disc.eval_batch(input_rows, tolerate_failure=True)

    def _differentiate(self, inputs: Mapping[str, ndarray],
                       outputs: Mapping[str, ndarray]) -> Dict[str, Dict[str, ndarray]]:
        return self.disc.differentiate(inputs)
