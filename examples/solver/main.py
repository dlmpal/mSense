from msense.api import *

from disciplines import *

x1 = Variable("x1")
x2 = Variable("x2")
z = Variable("z")
y12 = Variable("y12")
y21 = Variable("y21")

values = {"x1": 3.0, "x2": 3, "y12": 1.0, "y21": 1.0, "z": 3.0}
disc1 = SimpleDisc("D1", [x1, z, y12], [y21], func1, dfunc1)
disc1.add_default_inputs(values)
disc2 = SimpleDisc("D2", [x2, z, y21], [y12], func2, dfunc2)
disc2.add_default_inputs(values)

solver = create_solver([disc1, disc2], "NewtonRaphson")
solver.solve(values)
solver.plot_convergence_history()
print(disc1.n_eval, disc2.n_eval, disc1.n_diff, disc2.n_diff)
