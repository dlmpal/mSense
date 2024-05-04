from typing import Dict, List

from msense.core.variable import Variable
from msense.core.discipline import Discipline


def get_couplings(disciplines: List[Discipline]) -> List[Variable]:
    """
    Get the coupling variables for a list of disciplines.

    Args:
        disciplines (List[Discipline]): The list of disciplines

    Returns:
        List[Variable]: The coupling variables
    """
    input_vars, output_vars = [], []
    for disc in disciplines:
        input_vars += (disc.input_vars)
        output_vars += (disc.output_vars)

    coupling_vars = []
    for out_var in output_vars:
        if out_var in input_vars and out_var not in coupling_vars:
            coupling_vars.append(out_var)

    return coupling_vars
