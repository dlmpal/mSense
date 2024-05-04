from typing import Dict, List

from numpy import ndarray, reshape
from scipy.sparse import dok_matrix, eye
from scipy.sparse.linalg import spsolve

from msense.core.constants import FLOAT_DTYPE
from msense.core.variable import Variable
from msense.utils.array_and_dict_utils import get_variable_list_size, array_to_dict_2d


class JacobianAssembler:
    """
    Assemble jacobian matrices for a set of disciplines.
    """

    def assemble_partial(self, input_vars: List[Variable], output_vars: List[Variable], partial: Dict[str, ndarray], as_residual: bool = False) -> dok_matrix:
        """
        Assemble partial derivatives from dictionary into a sparse matrix.

        Args:
            input_vars (List[Variable]): Input variables list.
            output_vars (List[Variable]): Output variables list.
            partial (Dict[str, ndarray]): Partial derivatives dictionary.
            residual (bool, optional): Whether to treat the output variables as residuals. Defaults to False.

        Returns:
            dok_matrix: Assembled matrix of partial derivatives.
        """
        # Sign is negative if the outputs are treated as residuals
        sign = 1.0
        if as_residual:
            sign = -1.0

        n_inputs = get_variable_list_size(input_vars)
        n_outputs = get_variable_list_size(output_vars)
        dfdx = dok_matrix((n_outputs, n_inputs), dtype=FLOAT_DTYPE)

        row_idx = 0
        for out_var in output_vars:
            col_idx = 0
            for in_var in input_vars:
                if out_var.name == in_var.name:
                    dfdx[row_idx: row_idx + out_var.size, col_idx: col_idx +
                         in_var.size] = eye(out_var.size, dtype=FLOAT_DTYPE)
                else:
                    if in_var.name in partial[out_var.name]:
                        dfdx[row_idx: row_idx + out_var.size, col_idx: col_idx +
                             in_var.size] = sign * partial[out_var.name][in_var.name]
                col_idx += in_var.size
            row_idx += out_var.size

        return dfdx

    def assemble_total(self, input_vars: List[Variable], output_vars: List[Variable], coupling_vars: List[Variable], partial: Dict[str, ndarray]) -> Dict[str, Dict[str, ndarray]]:
        """
        Assembles the total derivatives of the outputs w.r.t the inputs, given the partials. 
        If the size of inputs is larger than the size of the outputs, the adjoint method is used,
        else the direct.

        Args:
            input_vars (List[Variable]): Input variables list.
            output_vars (List[Variable]): Output variables list.
            coupling_vars (List[Variable]): Coupling variables list.
            partial (Dict[str, ndarray]): Partial derivatives dictionary.

        Returns:
            Dict[str, Dict[str, ndarray]]: Total derivatives dictionary.
        """
        n_inputs = get_variable_list_size(input_vars)
        n_outputs = get_variable_list_size(output_vars)

        # Assemble the partial derivative matrices
        dRdy = self.assemble_partial(
            coupling_vars, coupling_vars, partial, True)
        dRdx = self.assemble_partial(input_vars, coupling_vars, partial, True)
        dfdy = self.assemble_partial(
            coupling_vars, output_vars, partial, False)
        dfdx = self.assemble_partial(input_vars, output_vars, partial, False)

        # Compute the total derivatives, given by total = dfdy @ (-dRdy^-1 @ dRdx)
        # Adjoint
        if not n_inputs >= n_outputs:
            dfdy, dRdy, dRdx = dfdy.tocsr(), dRdy.tocsr(), dRdx.tocsr()
            dfdy, dRdy = dfdy.transpose(), dRdy.transpose()
            total = dfdx - spsolve(dRdy, dfdy).transpose() @ dRdx
        # Direct
        else:
            dfdy, dRdy, dRdx = dfdy.tocsr(), dRdy.tocsc(), dRdx.tocsc()
            total = dfdx - dfdy @ spsolve(dRdy, dRdx)

        # Convert the array to dictionary
        if type(total) != ndarray:
            total = total.toarray()
        total = reshape(total, (n_outputs, n_inputs))
        total = array_to_dict_2d(input_vars, output_vars, total)

        return total
