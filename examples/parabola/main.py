import numpy as np
from msense.api import *

create_logger()

x1 = Variable("x1", lb=2, ub=100)
x2 = Variable("x2", lb=3, ub=90)
y = Variable("y")


class Parabola(Discipline):
    def __init__(self):
        super().__init__("Parabola", [x1, x2], [y],
                         cache_policy=CachePolicy.FULL)

    def _eval(self) -> None:
        self._values["y"] = self._values["x1"]**2 + self._values["x2"]**2

    def _differentiate(self) -> None:
        self._jac["y"] = {"x1": 2 * self._values["x1"],
                          "x2": 2 * self._values["x2"]}


parabola = Parabola()
prob = create_opt_problem(type="SingleDiscipline", name="ParabolaOpt", disciplines=[parabola], design_vars=[
    x1, x2], objective=y, use_norm=True, cache_policy="full", cache_path="opt_problem")

result = prob.solve({"x1": np.array([50.0]), "x2": np.array([50.])})
prob.plot_objective_history()
print("Parabola evaluations: ", parabola.n_eval)
prob.save_cache(), parabola.save_cache()
