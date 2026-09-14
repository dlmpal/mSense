from typing import Dict, List, Mapping, Sequence, Tuple
import logging

from numpy import ndarray, ones, full, empty, concatenate, vstack, atleast_1d
from numpy import isinf, isneginf, isnan, inf, where, argmin
import matplotlib.pyplot as plt

from msense.core.constants import FLOAT_DTYPE
from msense.core.exceptions import DriverCapabilityError
from msense.core.variable import Variable
from msense.core.discipline import Discipline
from msense.opt.formulation import Objective, Constraint
from msense.opt.formulation import as_objectives, as_constraints
from msense.utils.array_and_dict_utils import copy_dict_1d
from msense.utils.array_and_dict_utils import concatenate_variable_bounds
from msense.utils.array_and_dict_utils import dict_to_array_2d
from msense.utils.array_and_dict_utils import denormalize_dict_1d, normalize_dict_1d
from msense.utils.array_and_dict_utils import normalize_dict_2d
from msense.opt.drivers.driver import Driver, DriverResult
from msense.opt.drivers.factory import create_driver

logger = logging.getLogger(__name__)


class OptProblem(Discipline):
    """
    Base optimization problem class.

    The problem owns the single canonical statement of the formulation, and every
    driver consumes it through the same small surface:

        n_obj / n_ineq / n_eq         problem dimensions
        objective_array(values)       F, in minimization form
        objective_jac_array(jac)      dF/dx, in minimization form
        constraint_residuals(values)  (g, h), with g <= 0 and h == 0
        constraint_bounds()           (cl, cu), the two-sided form
        design_bounds(use_norm)       (xl, xu)

    Drivers therefore never re-derive a constraint convention, and the sense of
    each objective is applied in exactly one place.
    """

    #: Objective value standing in for a failed evaluation. Deliberately large
    #: and finite rather than inf: population algorithms compute ideal/nadir
    #: points and statistics over F, and inf breaks them.
    failed_objective_value: float = 1e10

    #: Residual value standing in for a constraint whose evaluation failed.
    failed_constraint_violation: float = 1e10

    def __init__(self, name: str, design_vars: List[Variable],
                 objectives: Sequence[Objective], constraints: Sequence[Constraint] = None,
                 use_norm: bool = True, scalarize: bool = False,
                 driver: Driver = None, **cache_options) -> None:
        """
        Initialize the optimization problem.

        Args:
            name (str): Name by which the optimization problem is referenced.
            design_vars (List[Variable]): The design variables.
            objectives (Sequence[Objective]): The objectives. A bare Variable is taken
                to be a minimized Objective.
            constraints (Sequence[Constraint], optional): The constraints. A bare Variable
                is taken to be a Constraint, whose behaviour is given by its bounds.
            use_norm (bool, optional): Whether to normalize the design variables. Requires
                that all of them have finite bounds. Defaults to True.
            scalarize (bool, optional): Whether a driver that cannot handle multiple
                objectives may combine them by weighted sum. Defaults to False.
            driver (Driver, optional): The driver. Defaults to a ScipyDriver.
        """
        self.name = name
        self.design_vars = design_vars
        self.objectives = as_objectives(objectives)
        self.constraints = as_constraints(constraints)
        self.scalarize = scalarize

        if not self.objectives:
            raise ValueError(f"{self.name}: at least one objective is required.")

        # Normalization can be used only if
        # all design variables have finite bounds
        self.use_norm = use_norm
        for var in self.design_vars:
            if isinf(var.ub) or isneginf(var.lb):
                self.use_norm = False
                break

        # Initialize the underlying Discipline
        input_vars = self.design_vars
        output_vars = [obj.var for obj in self.objectives] + \
                      [con.var for con in self.constraints]
        super().__init__(name, input_vars, output_vars, **cache_options)

        # Driver
        self.driver = driver if driver is not None else create_driver(self)

        # Optimization history and the result of the last solve
        self.history: List[Dict[str, ndarray]] = []
        self.population_history: List[List[Dict[str, ndarray]]] = []
        self.result: DriverResult = None

    #
    # Problem dimensions
    #

    @property
    def n_design(self) -> int:
        return sum(var.size for var in self.design_vars)

    @property
    def n_obj(self) -> int:
        return sum(obj.var.size for obj in self.objectives)

    @property
    def n_ineq(self) -> int:
        return sum(con.n_rows for con in self.constraints if not con.is_equality)

    @property
    def n_eq(self) -> int:
        return sum(con.n_rows for con in self.constraints if con.is_equality)

    #
    # Canonical formulation
    #

    def objective_array(self, values: Mapping[str, ndarray]) -> ndarray:
        """
        The objective vector in minimization form, shape (n_obj,).

        A NaN objective (a failed evaluation) is mapped to failed_objective_value,
        so that the design is dominated by every valid one rather than poisoning
        the run.
        """
        rows = [obj.sense * atleast_1d(values[obj.name]) for obj in self.objectives]
        F = concatenate(rows).astype(FLOAT_DTYPE)
        return where(isnan(F), self.failed_objective_value, F)

    def objective_jac_array(self, jac: Mapping[str, Mapping[str, ndarray]]) -> ndarray:
        """
        The objective jacobian in minimization form, shape (n_obj, n_design).
        """
        rows = [obj.sense * dict_to_array_2d(self.design_vars, [obj.var], jac)
                for obj in self.objectives]
        return vstack(rows)

    def scalar_objective(self, values: Mapping[str, ndarray]) -> float:
        """
        The objectives reduced to a single number, for drivers that need one.
        A single objective is passed through; several are combined by weight.
        """
        F = self.objective_array(values)
        if F.size == 1:
            return float(F[0])
        return float(F @ self._objective_weights())

    def scalar_objective_jac(self, jac: Mapping[str, Mapping[str, ndarray]]) -> ndarray:
        """
        The gradient of scalar_objective, shape (n_design,).
        """
        J = self.objective_jac_array(jac)
        if J.shape[0] == 1:
            return J[0]
        return self._objective_weights() @ J

    def _objective_weights(self) -> ndarray:
        return concatenate([full(obj.var.size, obj.weight, FLOAT_DTYPE)
                            for obj in self.objectives])

    def constraint_residuals(self, values: Mapping[str, ndarray]) -> Tuple[ndarray, ndarray]:
        """
        The constraints as residuals: (g, h), with g <= 0 and h == 0 when satisfied.

        A NaN constraint (a failed evaluation) is reported as a large violation.
        """
        g_rows, h_rows = [], []
        for con in self.constraints:
            rows = con.residuals(
                atleast_1d(values[con.name]).astype(FLOAT_DTYPE))
            (h_rows if con.is_equality else g_rows).append(rows)

        g = concatenate(g_rows) if g_rows else empty(0, FLOAT_DTYPE)
        h = concatenate(h_rows) if h_rows else empty(0, FLOAT_DTYPE)

        violation = self.failed_constraint_violation
        return (where(isnan(g), violation, g), where(isnan(h), violation, h))

    def constraint_bounds(self) -> Tuple[ndarray, ndarray]:
        """
        The two-sided constraint bounds (cl, cu), for drivers that take them directly.

        These are bounds on the raw constraint values, and are deliberately never
        normalized: normalization scales design variables, not outputs.
        """
        cl, cu, _ = concatenate_variable_bounds(
            [con.var for con in self.constraints], use_normalization=False)
        return cl, cu

    def design_bounds(self, use_norm: bool = None) -> Tuple[ndarray, ndarray]:
        """
        The design variable bounds (xl, xu).
        """
        if use_norm is None:
            use_norm = self.use_norm
        xl, xu, _ = concatenate_variable_bounds(self.design_vars, use_norm)
        return xl, xu

    def is_feasible(self, values: Mapping[str, ndarray], tol: float = 1e-6) -> bool:
        """
        Whether a set of values satisfies every constraint to within tol.
        """
        g, h = self.constraint_residuals(values)
        return bool((g <= tol).all() and (abs(h) <= tol).all())

    #
    # Evaluation
    #

    def denormalize(self, design_vec: Mapping[str, ndarray]) -> Dict[str, ndarray]:
        """
        Convert a design vector from the driver's units to physical units.

        Normalization belongs to the driver, not to evaluation: only the
        optimizer works in a scaled design space, while the problem, its cache
        and its history are all stated in physical units. eval() therefore takes
        physical values, and drivers convert at their own boundary.
        """
        design_vec = copy_dict_1d(self.design_vars, design_vec)
        if self.use_norm:
            design_vec = denormalize_dict_1d(self.design_vars, design_vec)
        return design_vec

    def normalize(self, design_vec: Mapping[str, ndarray]) -> Dict[str, ndarray]:
        """
        Convert a design vector from physical units to the driver's units.
        """
        design_vec = copy_dict_1d(self.design_vars, design_vec)
        if self.use_norm:
            design_vec = normalize_dict_1d(self.design_vars, design_vec)
        return design_vec

    def normalize_jac(self, jac: Mapping[str, Mapping[str, ndarray]]) -> Dict[str, Dict[str, ndarray]]:
        """
        Apply the chain rule for the design variable normalization, turning
        partials with respect to physical design variables into partials with
        respect to the driver's scaled ones.
        """
        if not self.use_norm:
            return jac
        return normalize_dict_2d(self.input_vars, self.output_vars, jac)

    #
    # Solution
    #

    def update_history(self, values: Mapping[str, ndarray] = None,
                       population: Sequence[Mapping[str, ndarray]] = None) -> None:
        """
        Record one major driver iteration.

        Should be called at the end of each major driver/optimizer iteration.
        The driver supplies what it evaluated: a local driver passes the values
        at the current iterate, a population driver the whole generation. The
        problem keeps no record of an in-flight evaluation of its own.
        """
        recorded = self.input_vars + self.output_vars

        if population is not None:
            generation = [copy_dict_1d(recorded, member) for member in population]
            self.population_history.append(generation)

            # Track the best member so that the scalar history stays meaningful
            best = self._best_of(generation)
            if best is not None:
                self.history.append(best)
                logger.info(f"{self.name} - Generation: {self.driver.iter} - "
                            f"Population: {len(generation)} - "
                            f"Best objective: {self.objective_array(best)}")
            return

        if values is None:
            return

        entry = copy_dict_1d(recorded, values)
        self.history.append(entry)
        logger.info(f"{self.name} - Iteration: {self.driver.iter} - "
                    f"Objective: {self.objective_array(entry)}")

    def _best_of(self, members: Sequence[Mapping[str, ndarray]]) -> Dict[str, ndarray]:
        """
        Pick a representative best member: the feasible one with the lowest
        scalarized objective, or, if none are feasible, the least infeasible.
        """
        if not members:
            return None

        feasible = [m for m in members if self.is_feasible(m)]
        if feasible:
            return min(feasible, key=self.scalar_objective)

        def total_violation(m):
            g, h = self.constraint_residuals(m)
            return float(where(g > 0, g, 0).sum() + abs(h).sum())

        return min(members, key=total_violation)

    def solve(self, design_vec: Mapping[str, ndarray] = None) -> DriverResult:
        """
        Solve the optimization problem using the selected driver.

        Args:
            design_vec (Mapping[str, ndarray], optional): The starting design vector.
                Local drivers require one; population drivers use it, if given, to
                seed the initial population.

        Returns:
            DriverResult: The outcome, also stored as self.result.
        """
        # Reset history
        self.history, self.population_history = [], []

        # Check that the driver can handle this formulation
        self._check_driver()

        # Set driver callback
        self.driver.callback = self.update_history

        # Solve
        result = self.driver.solve(design_vec, self.use_norm)
        self.result = result

        if result.converged:
            logger.info(
                f"{self.name} has converged successfully in {self.driver.iter} iterations.")
        else:
            logger.warning(
                f"{self.name} has not converged in {self.driver.iter} iterations. "
                f"Related driver message: {result.message}")

        return result

    def _check_driver(self) -> None:
        """
        Verify that the driver supports what the formulation asks of it.
        """
        driver = self.driver

        if self.n_obj > 1 and not driver.supports_multi_objective:
            if not self.scalarize:
                raise DriverCapabilityError(
                    f"{self.name} has {self.n_obj} objectives, but driver "
                    f"{type(driver).__name__} handles only one. Use a driver that "
                    f"supports multiple objectives (e.g. a PymooDriver running NSGA2), "
                    f"or pass scalarize=True to combine them by weight.")
            logger.warning(
                f"{self.name}: {type(driver).__name__} cannot handle {self.n_obj} "
                f"objectives, so they are being combined by weighted sum. "
                f"The result is one point, not a Pareto front.")

        if (self.n_ineq or self.n_eq) and not driver.supports_nonlinear_constraints:
            raise DriverCapabilityError(
                f"{self.name} has constraints, but driver "
                f"{type(driver).__name__} does not support them.")

        if self.n_eq and not driver.supports_equality_constraints:
            raise DriverCapabilityError(
                f"{self.name} has equality constraints, but driver "
                f"{type(driver).__name__} does not support them.")

    #
    # Plotting
    #

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
            filename (str, optional): Where to save the plot. If None, defaults to the variable name.
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

    def plot_objective_history(self, show: bool = True, save: bool = False,
                               filename: str = None) -> None:
        """
        Plot the objective history. For a population driver, this is the best
        member of each generation.

        Args:
            show (bool, optional): Whether to show the plot. Defaults to True.
            save (bool, optional): Whether to save the plot. Defaults to False.
            filename (str, optional): Where to save the plot. If None, defaults to the problem name.
        """
        if not self.history:
            return

        x_label = "Generation No." if self.population_history else "Iteration No."

        fig, ax = plt.subplots(1, 1)
        for obj in self.objectives:
            for comp in range(obj.var.size):
                label = obj.name if obj.var.size == 1 else f"{obj.name}[{comp}]"
                ax.plot([i for i in range(len(self.history))],
                        [entry[obj.name][comp] for entry in self.history],
                        '-o', label=label)

        ax.grid()
        ax.set_xlim(-1, len(self.history))
        ax.set_xlabel(x_label)
        ax.set_ylabel("Objective")
        ax.set_title(f"{self.name} objective history")
        if self.n_obj > 1:
            ax.legend()

        if save:
            plt.savefig(fname=filename if filename else self.name)

        if show:
            plt.show()

    def plot_pareto_front(self, obj_x: Objective = None, obj_y: Objective = None,
                          show: bool = True, save: bool = False, filename: str = None) -> None:
        """
        Plot the non-dominated set of the last solve, for two of the objectives.

        Args:
            obj_x (Objective, optional): Objective on the x-axis. Defaults to the first.
            obj_y (Objective, optional): Objective on the y-axis. Defaults to the second.
            show (bool, optional): Whether to show the plot. Defaults to True.
            save (bool, optional): Whether to save the plot. Defaults to False.
            filename (str, optional): Where to save the plot. If None, defaults to the problem name.
        """
        if self.result is None or not self.result.front_values:
            logger.warning(
                f"{self.name}: no Pareto front to plot. The last solve either "
                f"has not run or used a single-objective driver.")
            return

        if len(self.objectives) < 2:
            logger.warning(f"{self.name}: a Pareto front needs two objectives.")
            return

        obj_x = obj_x if obj_x is not None else self.objectives[0]
        obj_y = obj_y if obj_y is not None else self.objectives[1]

        fig, ax = plt.subplots(1, 1)
        ax.plot([float(atleast_1d(v[obj_x.name])[0]) for v in self.result.front_values],
                [float(atleast_1d(v[obj_y.name])[0]) for v in self.result.front_values],
                'o', color="blue")
        ax.grid()
        ax.set_xlabel(obj_x.name)
        ax.set_ylabel(obj_y.name)
        ax.set_title(f"{self.name} Pareto front")

        if save:
            plt.savefig(fname=filename if filename else f"{self.name}_front")

        if show:
            plt.show()
