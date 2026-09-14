"""
Minimize y = x1^2 + x2^2 subject to g = x1 + x2 >= 10.

Demonstrates the formulation surface and swapping drivers underneath it.
"""
from pathlib import Path

import numpy as np
from msense.api import *

create_logger()

x1 = Variable("x1", lb=2, ub=100)
x2 = Variable("x2", lb=3, ub=90)
y = Variable("y")
g = Variable("g", lb=10)


CACHE_DIR = Path(__file__).parent


class Parabola(Discipline):
    def __init__(self):
        super().__init__("Parabola", [x1, x2], [y, g],
                         cache_policy=CachePolicy.FULL,
                         cache_path=str(CACHE_DIR / "parabola_cache.json"))

    def _eval(self, inputs) -> dict:
        return {"y": inputs["x1"]**2 + inputs["x2"]**2,
                "g": inputs["x1"] + inputs["x2"]}

    def _differentiate(self, inputs, outputs) -> dict:
        return {"y": {"x1": 2 * inputs["x1"], "x2": 2 * inputs["x2"]},
                "g": {"x1": 1.0, "x2": 1.0}}


parabola = Parabola()
prob = create_opt_problem(type="single_discipline",
                          name="ParabolaOpt",
                          disciplines=[parabola],
                          design_vars=[x1, x2],
                          objectives=[Objective(y)],
                          constraints=[Constraint(g)],
                          use_norm=True,
                          cache_policy=CachePolicy.FULL,
                          cache_path=str(CACHE_DIR / "problem_cache.json"))

# A gradient method, since this discipline provides analytic partials.
# Swap in method="COBYLA" for a gradient-free run, or
# create_driver(prob, type="pymoo_driver", algorithm="GA", pop_size=20)
# for a population-based one; nothing else in this script changes.
prob.driver = create_driver(prob, method="SLSQP", n_iter_max=100, tol=1e-9)

# Both caches persist across runs, so a second run of this script resolves
# every design vector it revisits without calling the discipline again.
parabola.load_cache(), prob.load_cache()

result = prob.solve({"x1": np.array([50.0]), "x2": np.array([50.0])})

print(f"converged: {result.converged} ({result.message})")
print(f"x1 = {result['x1'][0]:.6f}, x2 = {result['x2'][0]:.6f}")
print(f"y  = {result['y'][0]:.6f}, g = {result['g'][0]:.6f}")
print(f"iterations: {result.n_iter}, "
      f"discipline evaluations this run: {parabola.n_eval} "
      f"(the rest came from the cache)")

prob.plot_objective_history()

parabola.save_cache(), prob.save_cache()
