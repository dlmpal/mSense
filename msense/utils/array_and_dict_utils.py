from typing import Dict, List, Tuple

from numpy import zeros, ndarray
from numpy import atleast_1d, atleast_2d
from numpy.linalg import norm

from msense.core.constants import FLOAT_DTYPE
from msense.core.variable import Variable


def get_variable_list_size(vars: List[Variable]) -> int:
    size = 0
    for var in vars:
        size += var.size
    return size


def concatenate_variable_bounds(vars: List[Variable], use_normalization: bool = False) -> Tuple[ndarray, ndarray, ndarray]:
    """
    Get the bounds of a list of variables as a contiguous array.
    """
    n_vars = get_variable_list_size(vars)
    lb_arr = zeros(n_vars, FLOAT_DTYPE)
    ub_arr = zeros(n_vars, FLOAT_DTYPE)
    keep_feasible_arr = zeros(n_vars, bool)

    idx = 0
    for var in vars:
        lb, ub, keep_feasible = var.get_bounds_as_array(use_normalization)
        lb_arr[idx: idx + var.size] = lb
        ub_arr[idx: idx + var.size] = ub
        keep_feasible_arr[idx: idx + var.size] = keep_feasible
        idx += var.size

    return lb_arr, ub_arr, keep_feasible_arr


def verify_dict_1d(vars: List[Variable], values_dict: Dict[str, ndarray], dtype=FLOAT_DTYPE) -> Dict[str, ndarray]:
    """
    Check that entries with correct size and dtype exist for all vars
    in values_dict. Each entry is a 1d-numpy array

    Returns:
        Dict[str, ndarray]: The verified values dictionary.
    """
    for var in vars:
        if var.name not in values_dict:
            raise TypeError(f"Missing value for Variable {var.name}.")
        values_dict[var.name] = atleast_1d(values_dict[var.name]).astype(dtype)
        if var.size != values_dict[var.name].size:
            raise ValueError(
                f"Wrong size ({values_dict[var.name].size}), for Variable {var.name} of size {var.size}.")
    return values_dict


def copy_dict_1d(vars: List[Variable], values_dict: Dict[str, ndarray]) -> Dict[str, ndarray]:
    copy = {}
    for var in vars:
        if var.name in values_dict:
            copy[var.name] = atleast_1d(values_dict[var.name]).copy()
    return copy


def dict_to_array_1d(vars: List[Variable], values_dict: Dict[str, ndarray]) -> ndarray:
    n_vars = sum([var.size for var in vars])
    values_arr = zeros(n_vars, FLOAT_DTYPE)
    idx = 0
    for var in vars:
        values_arr[idx: idx + var.size] = values_dict[var.name]
        idx += var.size
    return values_arr


def array_to_dict_1d(vars: List[Variable], values_arr: ndarray) -> Dict[str, ndarray]:
    values_dict = {}
    idx = 0
    for var in vars:
        values_dict[var.name] = values_arr[idx: idx + var.size]
        idx += var.size
    return values_dict


def normalize_dict_1d(vars: List[Variable], values_dict: Dict[str, ndarray]) -> Dict[str, ndarray]:
    for var in vars:
        values_dict[var.name] = var.norm_values(values_dict[var.name])
    return values_dict


def denormalize_dict_1d(vars: List[Variable], values_dict: Dict[str, ndarray]) -> Dict[str, ndarray]:
    for var in vars:
        values_dict[var.name] = var.denorm_values(values_dict[var.name])
    return values_dict


def verify_dict_2d(input_vars: List[Variable], output_vars: List[Variable],
                   values_dict: Dict[str, Dict[str, ndarray]], dtype=FLOAT_DTYPE) -> Dict[str, Dict[str, ndarray]]:
    for out_var in output_vars:
        if out_var.name not in values_dict:
            raise TypeError(
                f"Missing entry for output Variable {out_var.name}.")
        for in_var in input_vars:
            if in_var.name not in values_dict[out_var.name]:
                raise TypeError(
                    f"For output Variable {out_var.name}, missing entry for input Variable {in_var.name}.")
            values_dict[out_var.name][in_var.name] = atleast_2d(
                values_dict[out_var.name][in_var.name]).astype(dtype)
            if values_dict[out_var.name][in_var.name].shape != (out_var.size, in_var.size):
                raise ValueError(
                    f"Wrong size ({values_dict[out_var.name][in_var.name].shape}) for entry ({out_var.name}, {in_var.name}) with size {(out_var.size, in_var.size)}.")
    return values_dict


def copy_dict_2d(input_vars: List[Variable], output_vars: List[Variable],
                 values_dict: Dict[str, ndarray]) -> Dict[str, Dict[str, ndarray]]:
    copy = {}
    for out_var in output_vars:
        if out_var.name in values_dict:
            copy[out_var.name] = {}
            for in_var in input_vars:
                if in_var.name in values_dict[out_var.name]:
                    copy[out_var.name][in_var.name] = atleast_2d(
                        values_dict[out_var.name][in_var.name]).copy()
    return copy


def dict_to_array_2d(input_vars: List[Variable], output_vars: List[Variable],
                     values_dict: Dict[str, Dict[str, ndarray]], flatten: bool = False) -> ndarray:
    n_in_vars = sum([var.size for var in input_vars])
    n_out_vars = sum([var.size for var in output_vars])
    values_arr = zeros((n_out_vars, n_in_vars), FLOAT_DTYPE)
    row_idx = 0
    for out_var in output_vars:
        col_idx = 0
        for in_var in input_vars:
            values_arr[row_idx: row_idx + out_var.size, col_idx: col_idx +
                       in_var.size] = values_dict[out_var.name][in_var.name]
            col_idx += in_var.size
        row_idx += out_var.size
    if flatten:
        return values_arr.reshape(-1)
    return values_arr


def array_to_dict_2d(input_vars: List[Variable], output_vars: List[Variable], values_arr: ndarray) -> Dict[str, Dict[str, ndarray]]:
    values_dict: Dict[str, Dict[str, ndarray]] = {}
    row_idx = 0
    for out_var in output_vars:
        col_idx = 0
        values_dict[out_var.name] = {}
        for in_var in input_vars:
            values_dict[out_var.name][in_var.name] = values_arr[row_idx: row_idx +
                                                                out_var.size, col_idx: col_idx + in_var.size]
            col_idx += in_var.size
        row_idx += out_var.size
    return values_dict


def normalize_dict_2d(input_vars: List[Variable], output_vars: List[Variable], values_dict: Dict[str, Dict[str, ndarray]]):
    for out_var in output_vars:
        for in_var in input_vars:
            values_dict[out_var.name][in_var.name] = in_var.norm_grad(
                values_dict[out_var.name][in_var.name])
    return values_dict


def denormalize_dict_2d(input_vars: List[Variable], output_vars: List[Variable],
                        values_dict: Dict[str, Dict[str, ndarray]]):
    for out_var in output_vars:
        for in_var in input_vars:
            values_dict[out_var.name][in_var.name] = in_var.denorm_grad(
                values_dict[out_var.name][in_var.name])
    return values_dict


def check_values_match(vars: List[Variable], original: Dict[str, ndarray] = {},
                       test: Dict[str, ndarray] = {}, tol: float = 1e-9) -> bool:
    if not original or not test:
        return False

    for var in vars:
        err = (norm(original[var.name] - test[var.name]
                    ) / (1.0 + norm(test[var.name])))
        if err >= tol:
            return False

    return True
