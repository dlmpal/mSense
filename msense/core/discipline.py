from typing import Dict
from typing import List
from enum import Enum
import logging

from numpy import ndarray

from msense.core.constants import FLOAT_DTYPE, COMPLEX_DTYPE
from msense.core.variable import Variable
from msense.cache.cache import CachePolicy
from msense.cache.factory import CacheType, create_cache
from msense.utils.array_and_dict_utils import verify_dict_1d, verify_dict_2d
from msense.utils.array_and_dict_utils import copy_dict_1d, copy_dict_2d
from msense.utils.jac_utils import forward_finite_difference_approx
from msense.utils.jac_utils import central_finite_difference_approx
from msense.utils.jac_utils import complex_step_approx
from msense.utils.jac_utils import initialize_dense_jac

logger = logging.getLogger(__name__)


class Discipline:
    """
    Base discipline class.
    """
    class DiffMethod(str, Enum):
        """
        The method by which the discipline is differentiated.
        """
        ANALYTIC = "analytic"
        FINITE_DIFFERENCE = "finite_difference",
        CENTRAL_FINITE_DIFFERENCE = "central_finite_difference"
        COMPLEX_STEP = "complex_step"

    class DiffPolicy(Enum):
        """
        Whether to evaluate a set of values before differentiating.
        """
        ALWAYS = True
        NEVER = False

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
            dinput_vars (List[Variable], optional): List of input variables w.r.t compute partials. Defaults to None.
            doutput_vars (List[Variable], optional): List of output variables for which partials are computed. Defaults to None.
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

        # Number of evaluations and differentiations
        self.n_eval, self.n_diff = 0, 0

        # Differentiation method, policy and approximation step (if required)
        self._diff_method = self.DiffMethod.ANALYTIC
        self._diff_policy = self.DiffPolicy.ALWAYS
        self._eps: float = 1e-6

        # Default evaluation inputs
        self._default_inputs: Dict[str, ndarray] = {}

        # Latest evaluation values
        self._values: Dict[str, ndarray] = {}

        # Latest jacobian
        self._jac: Dict[str, Dict[str, ndarray]] = {}

        # Whether the disciplone is undergoing jacobian approximation
        self._approximating_jac: bool = False

        # The datatype used for floating-point arithmetic
        self._dtype = FLOAT_DTYPE

    def __repr__(self) -> str:
        return self.name

    def get_input_values(self) -> Dict[str, ndarray]:
        """
        Get a copy of the current input values.
        """
        return copy_dict_1d(self.input_vars, self._values)

    def get_output_values(self) -> Dict[str, ndarray]:
        """
        Get a copy of the current output values.
        """
        return copy_dict_1d(self.output_vars, self._values)

    def get_values(self) -> Dict[str, ndarray]:
        """
        Get a copy of the current input and output values.
        """
        values = self.get_input_values()
        values.update(self.get_output_values())
        return values

    def get_default_inputs(self) -> Dict[str, ndarray]:
        """
        Get a copy of the default input values.
        """
        return copy_dict_1d(self.input_vars, self._default_inputs)

    def add_default_inputs(self, input_values: Dict[str, ndarray]) -> None:
        """
        Update the default input values.
        """
        self._default_inputs.update(
            copy_dict_1d(self.input_vars, input_values))

    def get_jac(self) -> Dict[str, Dict[str, ndarray]]:
        """
        Get a copy of the current jacobian.
        """
        return copy_dict_2d(self.dinput_vars, self.doutput_vars, self._jac)

    def _load_cache_entry_outputs(self) -> bool:
        """
        Check if a cache entry exists for the current input values.
        If yes, update the output values.

        Returns:
            bool: Whether an entry was found.
        """
        entry_exists = False
        if self.cache is not None:
            output_values, _ = self.cache.load_entry(self._values)
            if output_values:
                self._values.update(output_values)
                entry_exists = True
        return entry_exists

    def _add_cache_entry_outputs(self) -> None:
        """
        Add a cache entry for the current outputs values.
        """
        if self.cache is not None:
            self.cache.add_entry(self._values, self._values, None)

    def _load_cache_entry_jac(self) -> bool:
        """
        Check if a cache entry exists for the current input values.
        If yes, update the jacobian.

        Returns:
            bool: Whether an entry was found.
        """
        entry_exists = False
        if self.cache is not None:
            _, jac = self.cache.load_entry(self._values)
            if jac:
                self._jac.update(jac)
                entry_exists = True
        return entry_exists

    def _add_cache_entry_jac(self) -> None:
        """
        Add a cache entry for the current jacobian.
        """
        if self.cache is not None:
            self.cache.add_entry(self._values, None, self._jac)

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

    def _sanitize_inputs(self, input_values: Dict[str, ndarray]):
        """
        Sanitize the inputs:
        * If no value is provided for a variable,
        try to use the default value.
        * Check that no values are missing and that the sizes
        are correct.

        """
        if input_values is None:
            input_values = {}
        self._values.update(self.get_default_inputs())
        self._values.update(copy_dict_1d(self.input_vars, input_values))

        try:
            self._values = verify_dict_1d(
                self.input_vars, self._values, self._dtype)
        except Exception as e:
            logger.error(f"{self.name}: {e}")

    def _eval(self) -> None:
        """
        Update the values for the output variables
        * self._values should be updated here.
        """
        raise NotImplementedError

    def eval(self, input_values: Dict[str, ndarray] = None) -> Dict[str, ndarray]:
        """
        Execute the discipline for the given inputs.
        * If a cache exists, the outputs are cached.
        * The default values are updated by the current inputs, if evaluation is succeeds.

        Args:
            input_values (Dict[str, ndarray], optional): Input values for each variable. If not provided, try to use the defaults.

        Returns:
            Dict[str, ndarray]: The output values.
        """
        # Reset values and jacobian
        self._values, self._jac = {}, {}

        # Sanitize the inputs
        self._sanitize_inputs(input_values)

        # Check if corresponding cache entry exists
        # Else, evaluate using the given inputs
        entry_exists = self._load_cache_entry_outputs()
        if entry_exists == False:
            self._eval()

        # Verify the outputs
        try:
            self._values = verify_dict_1d(
                self.output_vars, self._values, self._dtype)
        except Exception as e:
            logger.error(f"{self.name}: {e}")

        # Increment evaluation counter
        if entry_exists == False:
            self.n_eval += 1

        # Update cache and default inputs
        if self._approximating_jac == False:
            self.add_default_inputs(self.get_input_values())
            if entry_exists == False:
                self._add_cache_entry_outputs()

        return self.get_output_values()

    def set_jacobian_approximation(self, method: DiffMethod = DiffMethod.FINITE_DIFFERENCE, eps: float = 1e-4) -> None:
        """
        Setup the jacobian approximation.
        """
        if method not in [self.DiffMethod.FINITE_DIFFERENCE,
                          self.DiffMethod.CENTRAL_FINITE_DIFFERENCE,
                          self.DiffMethod.COMPLEX_STEP]:
            logger.error(
                f"{self.name}: {method} is not a valid jacobian approximation method.")
        else:
            self._diff_method, self._eps = method, eps
            self._diff_policy = self.DiffPolicy.ALWAYS

    def _init_jacobian(self) -> None:
        """
        Initialize the jacobian.
        """
        self._jac = initialize_dense_jac(self.dinput_vars, self.doutput_vars)

    def _approximate_jacobian(self) -> None:
        """
        Approximate the jacobian using finite-differencing or the complex-step method.
        """
        self._approximating_jac = True
        # Save the current input/output values
        input_values = self.get_input_values()
        output_values = self.get_output_values()

        # Approximate jacobian
        # Finite-differences
        if self._diff_method == self.DiffMethod.FINITE_DIFFERENCE:
            self._jac = forward_finite_difference_approx(self.eval, self.dinput_vars, self.doutput_vars,
                                                         self.get_input_values(), self.get_output_values(), self._jac, self._eps)

        # Central finite-differences
        if self._diff_method == self.DiffMethod.CENTRAL_FINITE_DIFFERENCE:
            self._jac = central_finite_difference_approx(self.eval, self.dinput_vars, self.doutput_vars,
                                                         self.get_input_values(), self.get_output_values(), self._jac, self._eps)

        # Complex-step
        if self._diff_method == self.DiffMethod.COMPLEX_STEP:
            self._dtype = COMPLEX_DTYPE
            self._jac = complex_step_approx(self.eval, self.dinput_vars, self.doutput_vars,
                                            self.get_input_values(), self._jac, self._eps)
            self._dtype = FLOAT_DTYPE

        # Reset the values
        self._values.update(input_values)
        self._values.update(output_values)
        self._approximating_jac = False

    def _differentiate(self) -> None:
        """
        Update the values for the jacobian.
        * self._jac should be updated here.
        """
        raise NotImplementedError

    def differentiate(self, input_values: Dict[str, ndarray] = None) -> Dict[str, Dict[str, ndarray]]:
        """
        Differentiate the discipline for a given set of input values.
        * If evaluation is required before differentiation (self._diff_policy = ALWAYS), self.eval() is called first with the same inputs.
        * If the jacobian is approximated, evaluation is always performed first.
        * If a cache exists, the jacobian is cached.

        Args:
            input_values (Dict[str, ndarray], optional): Input values for each variable. If not provided, try to use the defaults.

        Returns:
            Dict[str, ndarray]: The jacobian.

        """
        # Reset values and jacobian
        self._values, self._jac = {}, {}

        # If approximating, enforce evaluation
        if self._diff_method != self.DiffMethod.ANALYTIC:
            self._diff_policy = self.DiffPolicy.ALWAYS

        # If required, evaluate first
        # otherwise, just sanitize the inputs
        if self._diff_policy == self.DiffPolicy.ALWAYS:
            self.eval(input_values)
        else:
            self._sanitize_inputs(input_values)

        # Check if corresponding cache entry exists
        # Else, differentiate using the given inputs
        entry_exists = self._load_cache_entry_jac()
        if entry_exists == False:
            self._init_jacobian()
            if self._diff_method == self.DiffMethod.ANALYTIC:
                self._differentiate()
            else:
                self._approximate_jacobian()

        # Verify the jacobian and increment diff count
        try:
            self._jac = verify_dict_2d(
                self.dinput_vars, self.doutput_vars, self._jac, self._dtype)
        except Exception as e:
            logger.error(f"{self.name}: {e}")

        # Increment differentiation counter and update cache
        if entry_exists == False:
            self.n_diff += 1
            self._add_cache_entry_jac()

        return self.get_jac()
