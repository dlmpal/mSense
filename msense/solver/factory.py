from typing import List
from enum import Enum

from msense.core.discipline import Discipline
from msense.solver.solver import Solver
from msense.solver.nonlinear_jacobi import NonlinearJacobi
from msense.solver.nonlinear_gs import NonlinearGS
from msense.solver.newton_raphson import NewtonRaphson


class SolverType(str, Enum):
    NONLINEAR_GS = "NonlinearGS"
    NONLINEAR_JACOBI = "NonlinearJacobi"
    NEWTON_RAPHSON = "NewtonRaphson"


def create_solver(disciplines: List[Discipline], type: SolverType = SolverType.NONLINEAR_GS,
                  n_iter_max: int = 15, relax_fact: float = 1.0, tol: float = 0.0001, name: str = None) -> Solver:

    kwargs = {"disciplines": disciplines, "n_iter_max": n_iter_max,
              "relax_fact": relax_fact, "tol": tol}
    kwargs["name"] = name if name is not None else type

    if type == SolverType.NONLINEAR_GS:
        return NonlinearGS(**kwargs)
    if type == SolverType.NONLINEAR_JACOBI:
        return NonlinearJacobi(**kwargs)
    if type == SolverType.NEWTON_RAPHSON:
        return NewtonRaphson(**kwargs)
    else:
        raise ValueError(f"SolverType {type} is not available.")
