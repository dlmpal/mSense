from typing import List
from enum import Enum

from msense.core.variable import Variable
from msense.core.discipline import Discipline
from msense.opt.drivers.driver import Driver
from msense.opt.problems.opt_problem import OptProblem
from msense.opt.problems.single_discipline import SingleDiscipline
from msense.opt.problems.idf import IDF
from msense.opt.problems.mdf import MDF


class OptProblemType(str, Enum):
    SignleDiscipline = "SingleDiscipline"
    MDF = "MDF"
    IDF = "IDF"


def create_opt_problem(type: OptProblemType, disciplines: List[Discipline],
                       design_vars: List[Variable], objective: Variable, constraints: List[Variable] = None,
                       maximize_objective: bool = False, use_norm: bool = True, driver: Driver = None,
                       name: str = None, **options) -> OptProblem:

    kwargs = {"design_vars": design_vars, "objective": objective, "constraints": constraints,
              "maximize_objective": maximize_objective, "use_norm": use_norm, "driver": driver}
    kwargs["name"] = name if name is not None else type
    kwargs.update(options)

    if type == OptProblemType.SignleDiscipline:
        return SingleDiscipline(disciplines[0], **kwargs)
    elif type == OptProblemType.MDF:
        return MDF(disciplines, **kwargs)
    elif type == OptProblemType.IDF:
        return IDF(disciplines, **kwargs)
    else:
        raise ValueError(f"OptProblemType {type} is not available.")
