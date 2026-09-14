from typing import List, Sequence
from enum import Enum
import logging

from msense.core.variable import Variable
from msense.core.discipline import Discipline
from msense.opt.drivers.driver import Driver
from msense.opt.formulation import Objective, Constraint
from msense.opt.problems.opt_problem import OptProblem
from msense.opt.problems.single_discipline import SingleDiscipline
from msense.opt.problems.mdf import MDF

logger = logging.getLogger(__name__)


class OptProblemType(str, Enum):
    SINGLE_DISCIPLINE = "single_discipline"
    MDF = "mdf"
    IDF = "idf"
    CO = "co"


#: The multidisciplinary formulations still expect the previous, stateful
#: Discipline, in which evaluation left its values on the instance for the
#: caller to collect. They are kept in the tree but are not usable until they
#: are ported; see docs/optimisation-framework-plan.md, stage 11.
NOT_YET_PORTED = (OptProblemType.IDF, OptProblemType.CO)


def create_opt_problem(type: OptProblemType, disciplines: List[Discipline],
                       design_vars: List[Variable],
                       objectives: Sequence[Objective] = None,
                       constraints: Sequence[Constraint] = None,
                       use_norm: bool = True, scalarize: bool = False,
                       driver: Driver = None, name: str = None,
                       **options) -> OptProblem:
    """
    Create an optimization problem.

    Args:
        type (OptProblemType): The formulation to use.
        disciplines (List[Discipline]): The disciplines.
        design_vars (List[Variable]): The design variables.
        objectives (Sequence[Objective], optional): The objectives. A bare Variable
            is taken to be a minimized Objective.
        constraints (Sequence[Constraint], optional): The constraints. A bare Variable
            is taken to be a Constraint, whose behaviour is given by its bounds.
        use_norm (bool, optional): Whether to normalize the design variables. Defaults to True.
        scalarize (bool, optional): Whether a single-objective driver may combine
            several objectives by weighted sum. Defaults to False.
        driver (Driver, optional): The driver. Defaults to a ScipyDriver.
        name (str, optional): Problem name. Defaults to the formulation name.

    Returns:
        OptProblem: The problem.
    """
    type = OptProblemType(type)

    kwargs = {"design_vars": design_vars, "objectives": objectives,
              "constraints": constraints, "use_norm": use_norm,
              "scalarize": scalarize, "driver": driver}
    kwargs["name"] = name if name is not None else type.value
    kwargs.update(options)

    if type == OptProblemType.SINGLE_DISCIPLINE:
        return SingleDiscipline(disciplines[0], **kwargs)
    elif type == OptProblemType.MDF:
        return MDF(disciplines, **kwargs)

    if type in NOT_YET_PORTED:
        raise NotImplementedError(
            f"The '{type.value}' formulation has not been ported to the stateless "
            f"Discipline yet. It relied on a discipline holding the values of its "
            f"last evaluation, which no longer happens. Only "
            f"'{OptProblemType.SINGLE_DISCIPLINE.value}' is available.")

    raise ValueError(f"Unknown optimization problem type '{type}'.")
