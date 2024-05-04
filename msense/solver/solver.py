from typing import Dict, List
from enum import Enum
import logging

from numpy import ndarray, zeros
from numpy.linalg import norm
import matplotlib.pyplot as plt

from msense.core.constants import FLOAT_DTYPE
from msense.utils.array_and_dict_utils import verify_dict_1d
from msense.core.discipline import Discipline
from msense.utils.graph_utils import get_couplings

logger = logging.getLogger(__name__)


class Solver:
    """
    Base solver class 
    """

    class SolverStatus(Enum):
        NOT_CONVERGED = False
        CONVERGED = True

    def __init__(self, name: str, disciplines: List[Discipline], n_iter_max: int = 15,
                 relax_fact: float = 1.0, tol: float = 0.0001) -> None:
        """
        Initialize the solver.

        Args:
            name (str): Name by which the solver is referenced.
            disciplines (List[Discipline]): The list of disciplines to be included.
            n_iter_max (int, optional): Maximum solver iterations. Defaults to 15.
            relax_fact (float, optional): Solver relaxation factor. Defaults to 0.9.
            tol (float, optional): Residual tolerance. Defaults to 0.0001.
        """
        self.name = name
        self.disciplines = disciplines
        self.n_iter_max = n_iter_max
        self.relax_fact = relax_fact
        self.tol = tol

        # Coupling variables
        self.coupling_vars = get_couplings(self.disciplines)

        # History, iteration number and convergence status
        self.history, self.iter = [], 0
        self.status = self.SolverStatus.NOT_CONVERGED

        # Coupling variables values
        self._old_values: Dict[str, ndarray] = {}
        self._values: Dict[str, ndarray] = {}

    def _single_iteration(self):
        """
        Perform a single solver iteration.
        * self._values should be updated here
        * To be implemented by the subclasses.
        """
        raise NotImplementedError

    def _apply_relaxation(self) -> None:
        """
        Apply under/over relaxation
        """
        for var in self.coupling_vars:
            new, old = self._values[var.name], self._old_values[var.name]
            self._values[var.name] = self.relax_fact * \
                new + (1 - self.relax_fact) * old

    def _initialize_values(self, initial_coupling_values: Dict[str, ndarray]) -> None:
        """ Initialize the coupling variables values. 
        * Values not provided are obtained from the disciplines' default inputs, if they exist.
        * The old values are intiliazed to zero.

        Args:
            initial_coupling_values (Dict[str, ndarray]): The initial values provided by the caller.
        """
        self._values = {}
        if initial_coupling_values is None:
            initial_coupling_values = {}

        default_values = {}
        for disc in self.disciplines:
            default_values.update(disc.get_default_inputs())

        for var in self.coupling_vars:
            if var.name in initial_coupling_values:
                self._values[var.name] = initial_coupling_values[var.name]
            elif var.name in default_values:
                self._values[var.name] = default_values[var.name]

        try:
            self._values = verify_dict_1d(self.coupling_vars, self._values)
        except Exception as e:
            logger.error(f"{self.name}: {e}")

        for var in self.coupling_vars:
            self._old_values[var.name] = zeros(var.size, FLOAT_DTYPE)

    def _update_history(self) -> None:
        """
        Update residual metric history and convergence status
        """
        # Total residual metric
        metric = 0.0

        # Compute the residual for each coupling variable
        for var in self.coupling_vars:
            new, old = self._values[var.name], self._old_values[var.name]
            r = new - old
            metric += norm(r) / (1 + norm(new))
        self.history.append(metric)

        # Check for convergence
        if metric <= self.tol:
            self.status = self.SolverStatus.CONVERGED

        # Print iteration info
        logger.info(
            f"{self.name} - Iteration: {self.iter} - Residual: {metric}")

    def solve(self, initial_coupling_values: Dict[str, ndarray] = None) -> Dict[str, ndarray]:
        """
        Solve the nonlinear system for the coupling variables.

        Args:
            initial_coupling_values (Dict[str, ndarray], optional): The initial values for the coupling variables.
        Returns:
            Dict[str, ndarray]: The coupling variables values produced by the solver.
        """
        # Reset the solver
        self.history, self.iter = [], 0
        self.status = self.SolverStatus.NOT_CONVERGED
        self._initialize_values(initial_coupling_values)
        self._update_history()

        # Perform iterations
        while self.iter < self.n_iter_max:
            # If converged, stop iteration
            if self.status == self.SolverStatus.CONVERGED:
                break
            else:
                self.iter += 1

            # Update values
            self._old_values, self._values = self._values, {}
            self._single_iteration()

            # Apply the relaxation factor, if needed
            self._apply_relaxation()

            # Update residual metric
            self._update_history()

        if self.status == self.SolverStatus.CONVERGED:
            logger.info(
                f"{self.name} has converged in {self.iter} iterations.")
        else:
            metric = self.history[-1]
            logger.warn(
                f"{self.name} has reached the maximum number of iterations: ({self.n_iter_max}), "
                + f"but the residual: ({metric}), is above tolerance: ({self.tol})")

        return self._values

    def plot_convergence_history(self, show: bool = True, save: bool = False, filename: str = None) -> None:
        """
        Plot the residual metric history.

        Args:
            show (bool, optional): Whether to show the plot. Defaults to True.
            save (bool, optional): Whether to save the plot. Defaults to False.
            filename (str, optional): Where to save the plot. If None, defaults to the name of the solver.
        """
        if not self.history:
            return

        # Create the plot
        fig, ax = plt.subplots(1, 1)
        ax.semilogy([i for i in range(len(self.history))],
                    [total for total in self.history], '-o', color="blue")
        ax.axhline(self.tol, xmin=0, xmax=len(self.history), color="red")
        ax.set_title(f"{self.name} residual metric history")
        ax.set_ylabel("Residual metric")
        ax.set_xlabel("Iteration Νο.")
        ax.grid()

        # Save/show
        if save:
            if filename is None:
                filename = self.name
            plt.savefig(fname=filename)
        if show:
            plt.show()
