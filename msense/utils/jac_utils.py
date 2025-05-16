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


def forward_finite_difference_approx(func: Callable[[Dict[str, ndarray]], Dict[str, ndarray]],
                                     dinput_vars: List[Variable], doutput_vars: List[Variable],
                                     input_values: Dict[str, ndarray], output_values: Dict[str, ndarray] = None,
                                     jac: Dict[str, Dict[str, ndarray]] = None, eps=1e-6) -> Dict[str, Dict[str, ndarray]]:
    """
    Compute the jacobian of func using Finite-Differences
    """
    # Evaluate the function if the output values are not provided
    if output_values is None:
        output_values = func(input_values)

    # Initialize the jacobian, if one is not provided
    if jac is None:
        jac = initialize_dense_jac(dinput_vars, doutput_vars)

    # Loop over the input variables
    for in_var in dinput_vars:
        for i in range(in_var.size):
            # Compute the perturbation
            dx = eps * (1 + abs(input_values[in_var.name][i]))

            # Copy the input variable
            copy = input_values[in_var.name][i]

            # Perturb the input variable
            input_values[in_var.name][i] = copy + dx

            # Evaluate the function with the perturbed input values
            output_values_p = func(input_values)

            # Compute the jacobian entry
            for out_var in doutput_vars:
                jac[out_var.name][in_var.name][:, i] = (
                    output_values_p[out_var.name] - output_values[out_var.name]) / dx

            # Reset the perturbed value
            input_values[in_var.name][i] = copy

    return jac


def central_finite_difference_approx(func: Callable[[Dict[str, ndarray]], Dict[str, ndarray]],
                                     dinput_vars: List[Variable], doutput_vars: List[Variable],
                                     input_values: Dict[str, ndarray], output_values: Dict[str, ndarray] = None,
                                     jac: Dict[str, Dict[str, ndarray]] = None, eps=1e-6) -> Dict[str, Dict[str, ndarray]]:
    """
    Compute the jacobian of func using Finite-Differences
    """
    # Evaluate the function if the output values are not provided
    if output_values is None:
        output_values = func(input_values)

    # Initialize the jacobian, if one is not provided
    if jac is None:
        jac = initialize_dense_jac(dinput_vars, doutput_vars)

    # Loop over the input variables
    for in_var in dinput_vars:
        for i in range(in_var.size):
            # Compute the perturbation
            dx = eps * (1 + abs(input_values[in_var.name][i]))

            # Copy the input variable
            copy = input_values[in_var.name][i]

            # Perturb the input variable forward
            input_values[in_var.name][i] = copy + dx

            # Evaluate the function with the (forward) perturbed input values
            output_values_f = func(input_values)

            # Perturb the input variable backward
            input_values[in_var.name][i] = copy - dx

            # Evaluate the function with the (backward) perturbed input values
            output_values_b = func(input_values)

            # Compute the jacobian entry
            for out_var in doutput_vars:
                jac[out_var.name][in_var.name][:, i] = (
                    output_values_f[out_var.name] - output_values_b[out_var.name]) / (2 * dx)

            # Reset the perturbed value
            input_values[in_var.name][i] = copy

    return jac


def complex_step_approx(func: Callable[[Dict[str, ndarray]], Dict[str, ndarray]],
                        dinput_vars: List[Variable], doutput_vars: List[Variable],
                        input_values: Dict[str, ndarray],
                        jac: Dict[str, Dict[str, ndarray]], eps=1e-6) -> Dict[str, Dict[str, ndarray]]:
    """
    Compute the jacobian of func using the Complex-Step method
    """
    # Cast the input values to complex
    for in_var_name in input_values.keys():
        input_values[in_var_name] = input_values[in_var_name].astype(
            COMPLEX_DTYPE)

    # If required, initialize jac
    if jac is None:
        jac = initialize_dense_jac(dinput_vars, doutput_vars)

    # Loop over the input variables
    for in_var in dinput_vars:
        for i in range(in_var.size):

            # Perturb the input variable
            copy = input_values[in_var.name][i]
            input_values[in_var.name][i] += eps * 1j

            # Evaluate the function with the perturbed input values
            output_values_p = func(input_values)

            # Compute the jacobian entry
            for out_var in doutput_vars:
                jac[out_var.name][in_var.name][:, i] = imag(
                    output_values_p[out_var.name]) / eps

            # Reset the perturbed value
            input_values[in_var.name][i] = copy

    return jac
