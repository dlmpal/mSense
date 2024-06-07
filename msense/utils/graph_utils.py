from typing import Dict, List, Tuple

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


def separate_local_global(vars: List[Variable], disciplines: List[Discipline]) -> Tuple[Dict[Discipline, List[Variable]], List[Variable]]:
    """
    Separate local and global/shared variables, for a set of disciplines.
    A variable is considered local if it enters only 1 discipline, else it is global.
    """

    # Map each variable to one or more disciplines
    var_to_disc = {}
    for var in vars:
        var_to_disc[var] = []
        for disc in disciplines:
            if var in disc.input_vars:
                var_to_disc[var].append(disc)

    # Store the local variables for each discipline
    local_vars = {}
    for disc in disciplines:
        local_vars[disc] = []

    # Store the global varialbes in a contiguous list
    global_vars = []

    # Separate
    for var in var_to_disc:
        if len(var_to_disc[var]) == 1:
            disc = var_to_disc[var][0]
            local_vars[disc].append(var)
        else:
            global_vars.append(var)

    return local_vars, global_vars
