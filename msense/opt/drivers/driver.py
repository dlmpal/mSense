from typing import Callable, Dict, List, Mapping
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from numpy import ndarray

from msense.utils.array_and_dict_utils import array_to_dict_1d, dict_to_array_1d


@dataclass
class DriverResult:
    """
    The outcome of a driver run.

    A local driver fills in x/values and leaves the front empty. A multi-objective
    population driver fills in the front, and sets x/values to a representative
    member of it.
    """
    converged: bool
    message: str = ""

    #: The best, or chosen, design vector, in physical units.
    x: Dict[str, ndarray] = None
    #: Its objective and constraint values.
    values: Dict[str, ndarray] = None

    #: The non-dominated design vectors, for a multi-objective run.
    front_x: List[Dict[str, ndarray]] = None
    #: Their objective and constraint values.
    front_values: List[Dict[str, ndarray]] = None

    #: Counters.
    n_eval: int = 0
    n_iter: int = 0

    def __getitem__(self, name: str) -> ndarray:
        """
        Allow the result to be read like the values dictionary it replaces.
        """
        if self.x is not None and name in self.x:
            return self.x[name]
        if self.values is not None and name in self.values:
            return self.values[name]
        raise KeyError(name)

    @property
    def has_front(self) -> bool:
        return bool(self.front_values)


class Driver(ABC):
    """
    Base driver class.

    A driver takes a formulated OptProblem and runs an optimizer against it. The
    class attributes below declare what the driver can do; OptProblem.solve checks
    them against the formulation before running, so that an unsupported problem
    fails with an explanation instead of a wrong answer.
    """

    #: Whether the driver can optimize several objectives at once.
    supports_multi_objective: bool = False
    #: Whether the driver can handle nonlinear constraints.
    supports_nonlinear_constraints: bool = True
    #: Whether the driver can handle equality constraints.
    supports_equality_constraints: bool = True
    #: Whether the driver needs gradients.
    requires_gradients: bool = False
    #: Whether the driver evaluates a population per iteration.
    is_population_based: bool = False
    #: Whether a failed evaluation can be absorbed rather than ending the run.
    tolerates_failed_evaluations: bool = False

    def __init__(self, problem, n_iter_max: int = 10, tol: float = 1e-6,
                 callback: Callable = None, **options):
        """
        Initialize the driver.

        Args:
            problem (OptProblem): The problem to solve.
            n_iter_max (int, optional): Maximum major iterations (generations, for a
                population driver). Defaults to 10.
            tol (float, optional): Convergence tolerance. Defaults to 1e-6.
            callback (Callable, optional): Called at the end of each major iteration.
        """
        self.prob = problem
        self.n_iter_max = n_iter_max
        self.tol = tol
        self.callback = callback
        self.iter = 0
        self.options: Dict[str, any] = dict(options)

    #
    # The boundary between the optimizer's design space and the problem's
    #
    # The optimizer works in whatever space the problem's normalization defines;
    # the problem, its cache and its history are in physical units. Every
    # conversion happens here, at the one place that spans both, rather than
    # inside the problem's evaluation.
    #

    def design_dict(self, x: ndarray) -> Dict[str, ndarray]:
        """
        Convert an optimizer design vector into physical named values.
        """
        return self.prob.denormalize(
            array_to_dict_1d(self.prob.design_vars, x))

    def design_array(self, design_vec: Mapping[str, ndarray]) -> ndarray:
        """
        Convert physical named values into an optimizer design vector.
        """
        return dict_to_array_1d(self.prob.design_vars,
                                self.prob.normalize(design_vec))

    def evaluate(self, x: ndarray) -> Dict[str, ndarray]:
        """
        Evaluate the problem at an optimizer design vector.
        """
        return self.prob.eval(
            self.design_dict(x),
            tolerate_failure=self.tolerates_failed_evaluations)

    def differentiate(self, x: ndarray) -> Dict[str, Dict[str, ndarray]]:
        """
        Differentiate the problem at an optimizer design vector, returning
        partials with respect to the optimizer's design variables.
        """
        return self.prob.normalize_jac(
            self.prob.differentiate(self.design_dict(x)))

    def starting_array(self, x0: Mapping[str, ndarray]) -> ndarray:
        """
        The starting design vector, in the optimizer's units.

        Falls back to the problem's default inputs.
        """
        if not x0:
            x0 = self.prob.get_default_inputs()
        if not x0:
            raise ValueError(
                f"{type(self).__name__} requires a starting design vector.")
        return self.design_array(x0)

    def _callback(self, **kwargs) -> None:
        if self.callback is not None:
            self.callback(**kwargs)
        self.iter += 1

    @abstractmethod
    def solve(self, x0: Mapping[str, ndarray], use_norm: bool) -> DriverResult:
        """
        Run the optimizer.

        Args:
            x0 (Mapping[str, ndarray]): The starting design vector, in physical units.
                May be None for a driver that does not need one.
            use_norm (bool): Whether the design variables are normalized.

        Returns:
            DriverResult: The outcome.
        """
        ...
