from typing import Dict, List
import logging

from numpy import ndarray, isinf, isneginf
import matplotlib.pyplot as plt

from msense.core.variable import Variable
from msense.utils.array_and_dict_utils import copy_dict_1d
from msense.utils.array_and_dict_utils import denormalize_dict_1d, normalize_dict_2d
from msense.core.discipline import Discipline
from msense.opt.drivers.driver import Driver
from msense.opt.drivers.factory import create_driver

logger = logging.getLogger(__name__)


class OptProblem(Discipline):
    """
    Base optimization problem class.
    """

    def __init__(self, name: str, design_vars: List[Variable], objective: Variable, constraints: List[Variable] = None,
                 maximize_objective: bool = False, use_norm: bool = True, driver: Driver = None, **cache_options) -> None:
        """
        Initialize the multidicsiplinary optimization problem.

        Args:
            name (str): Name by which the optimization problem is referenced.
            design_vars (List[Variable]): The list of design variables.
            objective (Variable): The objective.
            constraints: (List[Variable], optional): The list of constraints. A constraint's behaviour is represented by its lower and upper bounds.
            maximize_objective (bool, optional): Whether to maximize the objective. Defaults to False.
            use_norm (bool, optional): Whether to use normalization. Defaults to True.
        """
        self.name = name
        self.design_vars = design_vars
        self.objective = objective
        self.constraints = constraints if constraints is not None else []
        self.max_obj = maximize_objective

        # Check if the objective is scalar
        if self.objective.size > 1:
            logger.warn(
                f"In {self.name}, the objective {self.objective.name} is not scalar (size = {self.objective.size}).")

        # Normalization can be used only if
        # all design variables have finite bounds
        self.use_norm = use_norm
        for var in self.design_vars:
            if isinf(var.ub) or isneginf(var.lb):
                self.use_norm = False
                break

        # Driver
        self.driver = driver
        if self.driver is None:
            self.driver = create_driver(self)

        # Initialize the underlying Discipline
        input_vars = self.design_vars
        output_vars = [self.objective] + self.constraints
        super().__init__(name, input_vars, output_vars, **cache_options)

        # Optimization history
        self.history = []

    def eval(self, design_vec: Dict[str, ndarray]):
        # Deormalize design vector, if needed
        if self.use_norm and self._approximating_jac is False:
            design_vec = denormalize_dict_1d(self.design_vars, design_vec)

        # Evaluate objective and constraints
        values = super().eval(design_vec)

        # Flip objective sign for maximization
        if self.max_obj:
            values[self.objective.name] *= -1

        return values

    def differentiate(self, design_vec: Dict[str, ndarray]):
        # Always evaluate before differentiating
        # Aside from enforcing consistency,
        # this is required when normalization is enabled
        if self._diff_policy != self.DiffPolicy.ALWAYS:
            self._diff_policy = self.DiffPolicy.ALWAYS

        # Compute the objective and constraints jacobian
        jac = super().differentiate(design_vec)

        # Normalize jacobian, if needed
        if self.use_norm:
            jac = normalize_dict_2d(
                self.input_vars, self.output_vars, jac)

        # Flip jacobian sign for maximization
        if self.max_obj:
            for var in self.design_vars:
                jac[self.objective.name][var.name] *= -1

        return jac

    def update_history(self) -> None:
        """
        Update the optimization problem history.
        Should be called at the end of each major driver/optimizer iteration.
        """
        entry = {}
        entry.update(copy_dict_1d(self.input_vars +
                     self.output_vars, self._values))
        self.history.append(entry)

        logger.info(
            f"{self.name} - Iteration: {self.driver.iter} - Objective: {self._values[self.objective.name][0]}")

    def solve(self, design_vec: Dict[str, ndarray]) -> Dict[str, any]:
        # Reset history
        self.history = []

        # Set driver callback
        self.driver.callback = self.update_history

        # Solve the problem using the selected driver
        converged, message = self.driver.solve(design_vec, self.use_norm)

        # Check convergence
        if converged:
            logger.info(
                f"{self.name} has converged successfully in {self.driver.iter} iterations.")
        else:
            logger.warn(
                f"{self.name} has not converged in {self.driver.iter} iterations. Related driver message: {message}")

        # Return the design variable values
        return self.get_input_values()

    def plot_variable_history(self, var: Variable, comp: int = 0,
                              show_lb: bool = False,  show_ub: bool = False,
                              y_label: str = None, title: str = None,
                              show: bool = True, save: bool = False, filename: str = None) -> None:
        """
        Plot the history for a specific variable (and component).

        Args:
            var (Variable): Variable
            comp (int, optional): Variable component (between 0 and var.size-1). Defaults to 0.
            show_lb (bool, optional): Whether to plot the lower bound. Defaults to False.
            show_ub (bool, optional): Whether to plot the upper bound. Defaults to False.
            y_label (str, optional): Plot y-axis label. If None, defaults to variable name.
            title (str, optional): Plot title. If None, defaults to variable name.
            show (bool, optional): Whether to show the plot. Defaults to True.
            save (bool, optional): Whether to save the plot. Defaults to False.
            filename (str, optional): Where to save the plot. If None, defaults to the name of the optimization problem.
        """
        if not self.history:
            return

        if var not in self.input_vars + self.output_vars:
            raise ValueError(
                f"Variable {var.name} not found in optimization problem {self.name}")

        if comp < 0 or comp >= var.size:
            raise IndexError(
                f"Component {comp} for variable {var.name} is either < 0 or >= variable size ({var.size})")

        fig, ax = plt.subplots(1, 1)

        ax.plot([i for i in range(len(self.history))],
                [entry[var.name][comp] for entry in self.history], '-o', color='blue')

        if show_lb and not isneginf(var.lb):
            ax.hlines([var.lb], xmin=-1,
                      xmax=len(self.history), colors=['red'])

        if show_ub and not isinf(var.ub):
            ax.hlines([var.ub], xmin=-1,
                      xmax=len(self.history), colors=['red'])

        ax.grid()
        ax.set_xlim(-1, len(self.history))
        ax.set_xlabel("Iteration No.")

        if y_label is None:
            y_label = var.name
        ax.set_ylabel(y_label)

        if title is None:
            title = f"{var.name} history"
        ax.set_title(title)

        if save:
            if filename is None:
                filename = var.name
            plt.savefig(fname=filename)

        if show:
            plt.show()

    def plot_objective_history(self, show: bool = True, save: bool = False, filename: str = None) -> None:
        """
        Plot the objective history.

        Args:
            show (bool, optional): Whether to show the plot. Defaults to True.
            save (bool, optional): Whether to save the plot. Defaults to False.
            filename (str, optional): Where to save the plot. If None, defaults to the name of the optimization problem.
        """
        self.plot_variable_history(self.objective,
                                   y_label="Objective",
                                   title=f"{self.name} objective history",
                                   show=show, save=save, filename=filename)
        # if not self.history:
        #     return

        # # Create the plot
        # fig, ax = plt.subplots(1, 1)
        # ax.plot([i for i in range(len(self.history))],
        #         [self.history[i][self.objective.name] for i in range(len(self.history))], '-o', color="blue")
        # ax.set_title(f"{self.name} objective history")
        # ax.set_ylabel(f"Objective")
        # ax.set_xlabel("Iteration No.")
        # ax.grid()

        # # Save/show
        # if save:
        #     if filename is None:
        #         filename = self.name
        #     plt.savefig(fname=filename)
        # if show:
        #     plt.show()
