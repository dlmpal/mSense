from typing import Dict, List, Mapping, Sequence, Union
from enum import Enum
import logging

from numpy import ndarray, full, nan, iscomplexobj

from msense.core.constants import FLOAT_DTYPE, COMPLEX_DTYPE
from msense.core.exceptions import EvaluationFailure
from msense.core.variable import Variable
from msense.cache.cache import CachePolicy
from msense.cache.factory import CacheType, create_cache
from msense.utils.array_and_dict_utils import verify_dict_1d, verify_dict_2d
from msense.utils.array_and_dict_utils import copy_dict_1d, check_values_match
from msense.utils.jac_utils import forward_finite_difference_approx
from msense.utils.jac_utils import central_finite_difference_approx
from msense.utils.jac_utils import complex_step_approx
from msense.utils.jac_utils import initialize_dense_jac

logger = logging.getLogger(__name__)

#: What one row of a batch computation yields: the outputs, or the failure.
BatchOutcome = Union[Dict[str, ndarray], EvaluationFailure]


class Discipline:
    """
    Base discipline class.

    A discipline is a pure function mapping input values to output values:
    evaluation carries no state between calls. Instance attributes hold
    *configuration* (name, variables, cache, default inputs), never the values
    of an in-flight evaluation. A single instance is therefore safe to evaluate
    concurrently.

    There are three operations, each a public method paired with a hook that
    subclasses implement:

        eval(input_values)        ->  _eval(inputs)                    required
        eval_batch(input_rows)    ->  _eval_batch(input_rows)          optional
        differentiate(input_vals) ->  _differentiate(inputs, outputs)  optional

    The public methods own everything that is not computation: filling in
    defaults, verifying sizes, consulting the cache, collapsing repeats,
    counting, and deciding whether a failure is fatal. The hooks are pure
    computation and are never called directly from outside. Nothing overrides a
    public method; a subclass that needs to transform values (a change of units,
    say) does it above this class, not by intercepting eval.

    _eval_batch exists only so that a discipline able to dispatch several
    evaluations at once -- an external solver across several processes, for
    instance -- can do so. Its default is a serial loop, and a discipline that
    has no such ability ignores it entirely.
    """

    class DiffMethod(str, Enum):
        """
        The method by which the discipline is differentiated.
        """
        ANALYTIC = "analytic"
        FINITE_DIFFERENCE = "finite_difference"
        CENTRAL_FINITE_DIFFERENCE = "central_finite_difference"
        COMPLEX_STEP = "complex_step"

    def __init__(self, name: str, input_vars: List[Variable], output_vars: List[Variable],
                 dinput_vars: List[Variable] = None, doutput_vars: List[Variable] = None,
                 cache_type: CacheType = CacheType.MEMORY, cache_policy: CachePolicy = CachePolicy.LATEST,
                 cache_tol: float = 1e-9, cache_path: str = None) -> None:
        """
        Initialize the discipline.

        Args:
            name (str): Name by which the discipline is referenced.
            input_vars (List[Variable]): List of input variables.
            output_vars (List[Variable]): List of output variables.
            dinput_vars (List[Variable], optional): List of input variables w.r.t compute partials. Defaults to input_vars.
            doutput_vars (List[Variable], optional): List of output variables for which partials are computed. Defaults to output_vars.
            cache_type (CacheType, optional): Type of cache. Defaults to CacheType.MEMORY.
            cache_policy (CachePolicy, optional): Caching policy. Defaults to CachePolicy.LATEST.
            cache_tol (float, optional): Cache tolerance. Defaults to 1e-9.
            cache_path (str, optional): Path to cache file. If None, the discipline name is used.
        """
        self.name: str = name
        self.input_vars: List[Variable] = input_vars
        self.output_vars: List[Variable] = output_vars
        self.dinput_vars: List[Variable] = dinput_vars
        if dinput_vars is None:
            self.dinput_vars = self.input_vars
        self.doutput_vars: List[Variable] = doutput_vars
        if doutput_vars is None:
            self.doutput_vars = self.output_vars

        # Discipline cache
        if cache_path is None:
            cache_path = self.name
        self.cache = create_cache(self.input_vars, self.output_vars,
                                  self.dinput_vars, self.doutput_vars,
                                  cache_type, cache_policy, cache_tol, cache_path)

        # Number of evaluations, differentiations and failed evaluations
        self.n_eval, self.n_diff, self.n_fail = 0, 0, 0

        # Differentiation method and approximation step (if required)
        self._diff_method = self.DiffMethod.ANALYTIC
        self._eps: float = 1e-6

        # Default evaluation inputs
        self._default_inputs: Dict[str, ndarray] = {}

    def __repr__(self) -> str:
        return self.name

    def get_default_inputs(self) -> Dict[str, ndarray]:
        """
        Get a copy of the default input values.
        """
        return copy_dict_1d(self.input_vars, self._default_inputs)

    def add_default_inputs(self, input_values: Mapping[str, ndarray]) -> None:
        """
        Update the default input values.
        """
        self._default_inputs |= copy_dict_1d(self.input_vars, input_values)

    def load_cache(self) -> None:
        """
        Try to load the discipline cache from file.
        """
        if self.cache is not None:
            self.cache.from_file()

    def save_cache(self) -> None:
        """
        Save the cache to file.
        """
        if self.cache is not None:
            self.cache.to_file()

    @property
    def _cache_tol(self) -> float:
        # check_values_match rejects everything at a tolerance of zero, so a
        # discipline with no cache still needs a value small enough that only an
        # exact repeat matches.
        return self.cache.tol if self.cache is not None else 1e-12

    def _resolve_dtype(self, values: Mapping[str, ndarray]):
        """
        The floating-point type for an evaluation: complex only if the caller
        supplied complex inputs, which the complex-step jacobian does.
        """
        for value in values.values():
            if iscomplexobj(value):
                return COMPLEX_DTYPE
        return FLOAT_DTYPE

    def _prepare_inputs(self, input_values: Mapping[str, ndarray]) -> Dict[str, ndarray]:
        """
        Build a private, verified input dictionary for one evaluation.

        Values not provided by the caller are taken from the default inputs.
        The result is a copy, so neither the caller's dictionary nor the
        defaults are modified by the evaluation.

        Raises:
            KeyError: If no value is available for an input variable.
            ValueError: If a provided value has the wrong size.
        """
        values = copy_dict_1d(self.input_vars, self._default_inputs)
        if input_values:
            values |= copy_dict_1d(self.input_vars, input_values)
        return verify_dict_1d(self.input_vars, values, self._resolve_dtype(values))

    def _record_outputs(self, inputs: Dict[str, ndarray], outcome: BatchOutcome,
                        tolerate_failure: bool, use_cache: bool, dtype=FLOAT_DTYPE) -> Dict[str, ndarray]:
        """
        Turn one computed outcome into verified outputs: count it, cache it, and
        decide whether a failure ends the run.

        Raises:
            EvaluationFailure: If the outcome is a failure and it is not tolerated.
        """
        if isinstance(outcome, EvaluationFailure):
            self.n_fail += 1
            if not tolerate_failure:
                raise outcome
            logger.warning(f"{self.name}: evaluation failed: {outcome}")
            outputs = {var.name: full(var.size, nan, dtype) for var in self.output_vars}
        else:
            outputs = copy_dict_1d(self.output_vars, verify_dict_1d(
                self.output_vars, outcome, dtype))

        self.n_eval += 1

        if use_cache:
            self.cache.add_entry(inputs, outputs, None)

        return outputs

    def _eval(self, inputs: Mapping[str, ndarray]) -> Dict[str, ndarray]:
        """
        Compute the output values for one set of input values.

        Args:
            inputs (Mapping[str, ndarray]): Value for each input variable.

        Returns:
            Dict[str, ndarray]: Value for each output variable.

        Raises:
            EvaluationFailure: If the outputs cannot be computed.
        """
        raise NotImplementedError

    def eval(self, input_values: Mapping[str, ndarray] = None,
             tolerate_failure: bool = False, use_cache: bool = True) -> Dict[str, ndarray]:
        """
        Evaluate the discipline for one set of inputs.

        Args:
            input_values (Mapping[str, ndarray], optional): Input values for each variable.
                Values not provided are taken from the default inputs.
            tolerate_failure (bool, optional): If True, an EvaluationFailure is caught and
                NaN is returned for every output instead of propagating. Defaults to False.
            use_cache (bool, optional): Whether to read from and write to the cache.
                Defaults to True.

        Returns:
            Dict[str, ndarray]: The output values.
        """
        inputs = self._prepare_inputs(input_values)
        dtype = self._resolve_dtype(inputs)
        use_cache = use_cache and self.cache is not None and dtype == FLOAT_DTYPE

        if use_cache:
            outputs, _ = self.cache.load_entry(inputs)
            if outputs:
                return outputs

        try:
            outcome = self._eval(inputs)
        except EvaluationFailure as e:
            outcome = e

        return self._record_outputs(inputs, outcome, tolerate_failure, use_cache, dtype)

    def _eval_batch(self, input_rows: Sequence[Mapping[str, ndarray]]) -> List[BatchOutcome]:
        """
        Compute the output values for several sets of input values.

        Override this, and only this, to dispatch evaluations concurrently. The
        rows are already prepared and already distinct: caching, de-duplication
        and counting are handled by eval_batch.

        Each entry of the returned list is either the outputs for that row or the
        EvaluationFailure it raised. Returning the failure rather than raising it
        keeps one bad design from discarding the results of the whole batch, and
        leaves the caller to decide what a failure means.

        Args:
            input_rows (Sequence[Mapping[str, ndarray]]): Input values per evaluation.

        Returns:
            List[BatchOutcome]: One outcome per row, in the order given.
        """
        outcomes = []
        for inputs in input_rows:
            try:
                outcomes.append(self._eval(inputs))
            except EvaluationFailure as e:
                outcomes.append(e)
        return outcomes

    def eval_batch(self, input_rows: Sequence[Mapping[str, ndarray]],
                   tolerate_failure: bool = False) -> List[Dict[str, ndarray]]:
        """
        Evaluate the discipline for many sets of inputs.

        Rows already in the cache are served from it, and rows repeating another
        row of the same batch are computed once. Only the remainder reach
        _eval_batch. All of that happens on the calling thread, so a concurrent
        _eval_batch needs no locking: the cache and the counters are never
        touched by a worker.

        Args:
            input_rows (Sequence[Mapping[str, ndarray]]): Input values per evaluation.
            tolerate_failure (bool, optional): See eval(). Defaults to False.

        Returns:
            List[Dict[str, ndarray]]: The output values, in the order given.
        """
        prepared = [self._prepare_inputs(row) for row in input_rows]
        outputs: List[Dict[str, ndarray]] = [None] * len(prepared)

        # Load what the cache already has, while keeping track of the inputs for which
        # the outputs have to be evaluated
        pending = []
        for i, inputs in enumerate(prepared):
            if self.cache is not None:
                cached, _ = self.cache.load_entry(inputs)
                if cached:
                    outputs[i] = cached
                    continue
            pending.append(i)

        # If all inputs are found in the cache, return early
        if not pending:
            return outputs

        # Collapse repeats within the batch onto one representative row.
        groups: List[List[int]] = []
        for i in pending:
            for group in groups:
                if check_values_match(self.input_vars, prepared[group[0]],
                                      prepared[i], self._cache_tol):
                    group.append(i)
                    break
            else:
                groups.append([i])

        # Evaluate one row per group
        outcomes = self._eval_batch([prepared[group[0]] for group in groups])
        for group, outcome in zip(groups, outcomes):
            computed = self._record_outputs(prepared[group[0]], outcome,
                                            tolerate_failure, True)
            for i in group:
                outputs[i] = {name: value.copy()
                              for name, value in computed.items()}

        return outputs

    def set_jacobian_approximation(self, method: DiffMethod = DiffMethod.FINITE_DIFFERENCE,
                                   eps: float = 1e-4) -> None:
        """
        Setup the jacobian approximation.
        """
        method = self.DiffMethod(method)
        if method == self.DiffMethod.ANALYTIC:
            logger.error(
                f"{self.name}: {method} is not a valid jacobian approximation method.")
            return
        self._diff_method, self._eps = method, eps

    def _differentiate(self, inputs: Mapping[str, ndarray],
                       outputs: Mapping[str, ndarray]) -> Dict[str, Dict[str, ndarray]]:
        """
        Compute the jacobian for one set of input values.

        The output values at the same point are supplied, because analytic
        partials commonly depend on them. They are guaranteed to belong to these
        inputs: differentiate() evaluates before differentiating.

        Args:
            inputs (Mapping[str, ndarray]): Value for each input variable.
            outputs (Mapping[str, ndarray]): Value for each output variable, at
                the same point.

        Returns:
            Dict[str, Dict[str, ndarray]]: d(output)/d(input), keyed [output][input].
        """
        raise NotImplementedError

    def _approximate_jacobian(self, inputs: Dict[str, ndarray],
                              outputs: Dict[str, ndarray] = None) -> Dict[str, Dict[str, ndarray]]:
        """
        Approximate the jacobian by finite differences or the complex-step method.

        Args:
            inputs (Dict[str, ndarray]): Value for each input variable.
            outputs (Dict[str, ndarray], optional): Value for each output variable at
                the base point. Saves the one-sided schemes from re-evaluating it.
        """
        jac = initialize_dense_jac(self.dinput_vars, self.doutput_vars)

        if self._diff_method == self.DiffMethod.FINITE_DIFFERENCE:
            return forward_finite_difference_approx(self.eval, self.dinput_vars, self.doutput_vars,
                                                    inputs, outputs, jac, self._eps)

        if self._diff_method == self.DiffMethod.CENTRAL_FINITE_DIFFERENCE:
            return central_finite_difference_approx(self.eval, self.dinput_vars, self.doutput_vars,
                                                    inputs, outputs, jac, self._eps)

        if self._diff_method == self.DiffMethod.COMPLEX_STEP:
            return complex_step_approx(self.eval, self.dinput_vars, self.doutput_vars,
                                       inputs, jac, self._eps)

        raise ValueError(f"{self.name}: unknown diff method {self._diff_method}")

    def differentiate(self, input_values: Mapping[str, ndarray] = None) -> Dict[str, Dict[str, ndarray]]:
        """
        Differentiate the discipline for one set of input values.

        The discipline is always evaluated at the point first: analytic partials
        commonly depend on the outputs, and a gradient must belong to the same
        point as the value it accompanies.

        Args:
            input_values (Mapping[str, ndarray], optional): Input values for each variable.
                Values not provided are taken from the default inputs.

        Returns:
            Dict[str, Dict[str, ndarray]]: The jacobian.
        """
        inputs = self._prepare_inputs(input_values)

        if self.cache is not None:
            _, jac = self.cache.load_entry(inputs)
            if jac:
                return jac

        outputs = self.eval(inputs)

        if self._diff_method == self.DiffMethod.ANALYTIC:
            jac = self._differentiate(inputs, outputs)
        else:
            jac = self._approximate_jacobian(inputs, outputs)

        jac = verify_dict_2d(self.dinput_vars, self.doutput_vars, jac)
        self.n_diff += 1

        if self.cache is not None:
            self.cache.add_entry(inputs, None, jac)

        return jac
