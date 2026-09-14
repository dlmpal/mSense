from typing import Dict, List, Mapping
import logging

from numpy import ndarray, vstack, asarray, atleast_2d, empty
from numpy.random import default_rng

from msense.core.constants import FLOAT_DTYPE
from msense.opt.drivers.driver import Driver, DriverResult


MSENSE_HAS_PYMOO = True
try:
    from pymoo.core.problem import Problem as PymooProblem
    from pymoo.core.callback import Callback as PymooCallback
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination
    from pymoo.operators.sampling.lhs import LHS
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.algorithms.soo.nonconvex.ga import GA
    from pymoo.algorithms.soo.nonconvex.de import DE
    from pymoo.algorithms.soo.nonconvex.pso import PSO
    from pymoo.util.ref_dirs import get_reference_directions
except ImportError:
    MSENSE_HAS_PYMOO = False
    PymooProblem = object
    PymooCallback = object

logger = logging.getLogger(__name__)


class PymooDriver(Driver):
    """
    This Driver subclass implements the functionality needed to solve an OptProblem
    using the population-based algorithms in pymoo.

    Unlike the local drivers, this one evaluates a whole generation per iteration
    and can return a Pareto front rather than a single point. It reaches the
    problem through OptProblem.eval_batch, so a discipline that dispatches its
    evaluations concurrently is used concurrently here without further work.
    """

    supports_multi_objective = True
    supports_nonlinear_constraints = True
    supports_equality_constraints = True
    requires_gradients = False
    is_population_based = True
    # A population algorithm samples the whole design box, so it will propose
    # designs the solver cannot converge. Those are penalized, not fatal.
    tolerates_failed_evaluations = True

    #: Algorithms that cannot handle constraints.
    UNCONSTRAINED_ONLY = ("PSO",)

    class _Problem(PymooProblem):
        """
        Adapts an OptProblem to the pymoo Problem interface.

        pymoo works with a design matrix and the arrays F/G/H; the conversion in
        both directions is exactly the canonical formulation surface of
        OptProblem, so there is no formulation logic here.
        """

        def __init__(self, driver, use_norm: bool):
            self.driver = driver
            self.prob = driver.prob
            xl, xu = self.prob.design_bounds(use_norm)

            # Named values for every design vector evaluated in this run, so that
            # the per-generation callback and the final front can be reported in
            # terms of variable names rather than raw arrays.
            self.seen: Dict[bytes, Dict[str, ndarray]] = {}

            super().__init__(n_var=xl.size,
                             n_obj=self.prob.n_obj,
                             n_ieq_constr=self.prob.n_ineq,
                             n_eq_constr=self.prob.n_eq,
                             xl=xl, xu=xu)

        def values_for(self, x: ndarray) -> Dict[str, ndarray]:
            """
            The named values recorded for a design vector, if it was evaluated.
            """
            return self.seen.get(asarray(x, FLOAT_DTYPE).tobytes())

        def _evaluate(self, X, out, *args, **kwargs):
            X = atleast_2d(X)
            rows = [self.driver.design_dict(x) for x in X]

            # One batch per generation: this is the seam a parallel discipline
            # plugs into.
            results = self.prob.eval_batch(rows, tolerate_failure=True)

            for x, physical, values in zip(X, rows, results):
                self.seen[asarray(x, FLOAT_DTYPE).tobytes()] = {
                    **physical, **values}

            out["F"] = vstack([self.prob.objective_array(v) for v in results])

            if self.prob.n_ineq:
                out["G"] = vstack(
                    [self.prob.constraint_residuals(v)[0] for v in results])

            if self.prob.n_eq:
                out["H"] = vstack(
                    [self.prob.constraint_residuals(v)[1] for v in results])

    class _Callback(PymooCallback):
        """
        Forwards each generation to the driver, so that OptProblem records a
        population history rather than an arbitrary single evaluation.
        """

        def __init__(self, driver, pymoo_problem):
            super().__init__()
            self.driver = driver
            self.pymoo_problem = pymoo_problem

        def notify(self, algorithm):
            population = []
            for x in algorithm.pop.get("X"):
                values = self.pymoo_problem.values_for(x)
                if values is not None:
                    population.append(values)
            self.driver._callback(population=population)

    def __init__(self, problem, algorithm: str = "NSGA2", pop_size: int = 40,
                 seed: int = 1, verbose: bool = False, termination=None,
                 sampling=None, **kwargs):
        """
        Initialize the driver.

        Args:
            problem (OptProblem): The problem to solve.
            algorithm (str, optional): One of NSGA2, NSGA3, GA, DE, PSO. Defaults to "NSGA2".
            pop_size (int, optional): Population size. Defaults to 40.
            seed (int, optional): Random seed. Defaults to 1.
            verbose (bool, optional): Whether pymoo prints its own progress table.
                mSense logs a line per generation regardless. Defaults to False.
            termination (optional): A pymoo termination object. Defaults to n_iter_max generations.
            sampling (optional): A pymoo sampling operator or an initial population
                array. Defaults to LHS seeded with the starting design vector, if given.
            n_iter_max (int, optional): Maximum number of generations.
        """
        if MSENSE_HAS_PYMOO is False:
            logger.error("Package pymoo is not available.")
            raise ImportError("pymoo is required by PymooDriver.")

        super().__init__(problem, **kwargs)
        self.algorithm = algorithm
        self.pop_size = pop_size
        self.seed = seed
        self.verbose = verbose
        self.termination = termination
        self.sampling = sampling

    def _initial_population(self, pymoo_problem, x0: ndarray) -> ndarray:
        """
        Build the initial population: Latin-hypercube samples over the design box,
        with the starting design vector as the first member.

        A converged baseline is expensive information and is usually the best
        design known at the outset, so it is worth carrying into generation zero.
        """
        if self.sampling is not None:
            return self.sampling

        if x0 is None:
            return LHS()

        # The samples are drawn here rather than by the algorithm, so the seed
        # has to be handed to the sampling explicitly: pymoo builds its own
        # generator when none is given, and does not consult numpy's global
        # one, so an unseeded draw here makes the whole run irreproducible.
        X = LHS()(pymoo_problem, self.pop_size,
                  random_state=default_rng(self.seed)).get("X")
        X[0] = x0
        return X

    def _create_algorithm(self, pymoo_problem, x0: ndarray):
        """
        Instantiate the requested pymoo algorithm.
        """
        name = self.algorithm.upper()
        sampling = self._initial_population(pymoo_problem, x0)
        options = dict(self.options)

        if name in self.UNCONSTRAINED_ONLY and (self.prob.n_ineq or self.prob.n_eq):
            logger.warning(
                f"pymoo algorithm {name} does not handle constraints; they will "
                f"be ignored by the search and only reported.")

        if name == "NSGA2":
            return NSGA2(pop_size=self.pop_size, sampling=sampling, **options)

        if name == "NSGA3":
            # NSGA3 partitions the objective space, so it needs reference directions
            ref_dirs = options.pop("ref_dirs", None)
            if ref_dirs is None:
                n_partitions = options.pop("n_partitions", 12)
                ref_dirs = get_reference_directions(
                    "das-dennis", self.prob.n_obj, n_partitions=n_partitions)
            return NSGA3(ref_dirs=ref_dirs, pop_size=self.pop_size,
                         sampling=sampling, **options)

        if name == "GA":
            return GA(pop_size=self.pop_size, sampling=sampling, **options)

        if name == "DE":
            return DE(pop_size=self.pop_size, sampling=sampling, **options)

        if name == "PSO":
            return PSO(pop_size=self.pop_size, sampling=sampling, **options)

        raise ValueError(
            f"Unknown pymoo algorithm '{self.algorithm}'. Available: "
            f"NSGA2, NSGA3, GA, DE, PSO.")

    def solve(self, x0: Mapping[str, ndarray], use_norm: bool) -> DriverResult:
        # A starting point is optional here: it seeds the initial population.
        x0_arr = self.design_array(x0) if x0 else None
        self.iter = 0

        pymoo_problem = self._Problem(self, use_norm)
        algorithm = self._create_algorithm(pymoo_problem, x0_arr)
        termination = self.termination
        if termination is None:
            termination = get_termination("n_gen", self.n_iter_max)

        result = minimize(pymoo_problem, algorithm, termination,
                          seed=self.seed, verbose=self.verbose,
                          callback=self._Callback(self, pymoo_problem))

        return self._build_result(result, pymoo_problem)

    def _build_result(self, result, pymoo_problem) -> DriverResult:
        """
        Convert a pymoo result into a DriverResult.
        """
        converged = result.X is not None
        message = "pymoo terminated"
        if not converged:
            message = ("pymoo found no solution; with constraints this usually "
                       "means no feasible design was located.")

        front_x, front_values = [], []
        if result.opt is not None:
            for x in atleast_2d(result.opt.get("X")):
                values = pymoo_problem.values_for(x)
                if values is None:
                    continue
                front_x.append(self.design_dict(x))
                front_values.append(values)

        # A representative point: the best member of the front
        best = self.prob._best_of(front_values) if front_values else None
        x_best = None
        if best is not None:
            x_best = {var.name: best[var.name] for var in self.prob.design_vars}

        return DriverResult(converged=converged,
                            message=message,
                            x=x_best,
                            values=best,
                            front_x=front_x or None,
                            front_values=front_values or None,
                            n_eval=self.prob.n_eval,
                            n_iter=self.iter)
