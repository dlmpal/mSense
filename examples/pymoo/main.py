"""
Two-objective optimization with pymoo's NSGA-II.

The structure mirrors an expensive external-solver problem: one discipline
producing several quantities of interest, two of which are in tension, plus a
constraint. Minimizing y while maximizing g is a genuine trade-off, so the
answer is a Pareto front rather than a point.
"""
import numpy as np
from msense.api import *

create_logger()

x1 = Variable("x1", lb=2, ub=100)
x2 = Variable("x2", lb=3, ub=90)
y = Variable("y")
g = Variable("g")
h = Variable("h", lb=20.0)


class Quantities(Discipline):
    def __init__(self):
        super().__init__("Quantities", [x1, x2], [y, g, h],
                         cache_policy=CachePolicy.FULL)

    def _eval(self, inputs) -> dict:
        # A design far out in x1 is declared unsolvable, the way a free-boundary
        # solve fails to converge for some coil layouts.
        if inputs["x1"][0] > 90.0:
            raise EvaluationFailure(f"no solution for x1={inputs['x1'][0]:.3f}")

        return {"y": inputs["x1"]**2 + inputs["x2"]**2,
                "g": inputs["x1"] + inputs["x2"],
                "h": 2 * inputs["x1"] + inputs["x2"]}


disc = Quantities()
prob = create_opt_problem(type="single_discipline",
                          name="TradeOff",
                          disciplines=[disc],
                          design_vars=[x1, x2],
                          objectives=[Objective(y),
                                      Objective(g, maximize=True)],
                          constraints=[Constraint(h)],
                          cache_policy=CachePolicy.FULL)

prob.driver = create_driver(prob, type="pymoo_driver",
                            algorithm="NSGA2",
                            pop_size=40,
                            n_iter_max=40,
                            seed=1)

# The starting point is optional for a population driver; when given, it becomes
# the first member of generation zero, so a known-good baseline is not thrown away.
result = prob.solve({"x1": np.array([10.0]), "x2": np.array([10.0])})

print(f"converged: {result.converged}")
print(f"evaluations: {result.n_eval}, generations: {result.n_iter}, "
      f"failures: {disc.n_fail}")
print(f"front size: {len(result.front_values)}")
print(f"{'y':>12} {'g':>12} {'h':>12}")
for values in sorted(result.front_values, key=lambda v: v["y"][0]):
    print(f"{values['y'][0]:12.3f} {values['g'][0]:12.3f} {values['h'][0]:12.3f}")

prob.plot_pareto_front()
prob.plot_objective_history()
