from typing import List, Dict, Callable

from numpy import ndarray, imag, zeros

from msense.core.constants import FLOAT_DTYPE, COMPLEX_DTYPE
from msense.core.variable import Variable


def initialize_dense_jac(dinput_vars: List[Variable], doutput_vars: List[Variable]):
    jac = {}
    for out_var in doutput_vars:
        jac[out_var.name] = {}
        for in_var in dinput_vars:
            jac[out_var.name][in_var.name] = zeros(
                (out_var.size, in_var.size), dtype=FLOAT_DTYPE)
    return jac


def finite_difference_approx(func: Callable[[Dict[str, ndarray]], Dict[str, ndarray]],
                             dinput_vars: List[Variable], doutput_vars: List[Variable],
                             input_values: Dict[str, ndarray], output_values: Dict[str, ndarray] = None,
                             jac: Dict[str, Dict[str, ndarray]] = None, eps=1e-6) -> Dict[str, Dict[str, ndarray]]:
    if output_values is None:
        output_values = func(input_values)
    if jac is None:
        jac = initialize_dense_jac(dinput_vars, doutput_vars)
    for in_var in dinput_vars:
        for i in range(in_var.size):
            # Perturb the i_th entry of the input variable
            temp = input_values[in_var.name][i].copy()
            input_values[in_var.name][i] += eps
            # Run the function with the perturbed input values
            output_values_p = func(input_values)
            # Compute the jacobian entry
            for out_var in doutput_vars:
                jac[out_var.name][in_var.name][:, i] = (
                    output_values_p[out_var.name] - output_values[out_var.name]) / eps
            # Reset the perturbed value
            input_values[in_var.name][i] = temp
    return jac


def complex_step_approx(func: Callable[[Dict[str, ndarray]], Dict[str, ndarray]],
                        dinput_vars: List[Variable], doutput_vars: List[Variable],
                        input_values: Dict[str, ndarray],
                        jac: Dict[str, Dict[str, ndarray]], eps=1e-6) -> Dict[str, Dict[str, ndarray]]:
    # Cast the input values to complex
    for in_var_name in input_values.keys():
        input_values[in_var_name] = input_values[in_var_name].astype(
            COMPLEX_DTYPE)
    # If required, initialize jac
    if jac is None:
        jac = initialize_dense_jac(dinput_vars, doutput_vars)
    for in_var in dinput_vars:
        for i in range(in_var.size):
            # Perturb the i_th entry of the input variable
            temp = input_values[in_var.name][i].copy()
            input_values[in_var.name][i] += eps * 1j
            # Run the function with the perturbed input values
            output_values_p = func(input_values)
            # Compute the jacobian entry
            for out_var in doutput_vars:
                jac[out_var.name][in_var.name][:, i] = imag(
                    output_values_p[out_var.name]) / eps
            # Reset the perturbed value
            input_values[in_var.name][i] = temp
    return jac
