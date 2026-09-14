from msense.api import *
import numpy as np


class SellarDiscipline1(Discipline):
    def __init__(self, z1: Variable, z2: Variable,
                 x1: Variable, y2: Variable,
                 y1: Variable, g1: Variable):

        super().__init__(name="SellarDiscipline1",
                         input_vars=[z1, z2, x1, y2],
                         output_vars=[y1, g1])

    def _eval(self, inputs):
        _z1 = inputs["z1"]
        _z2 = inputs["z2"]
        _x1 = inputs["x1"]
        _y2 = inputs["y2"]

        _y1 = np.sqrt(_z1**2 + _z2 + _x1 - 0.2 * _y2)

        return {"y1":  _y1,
                "g1": 3.16 - _y1**2}

    def _differentiate(self, inputs, outputs):
        _z1 = inputs["z1"]
        _y1 = outputs["y1"]

        jac = {}

        jac["y1"] = {"z1": _z1/_y1,
                     "z2": 1 / (2*_y1),
                     "x1": 1/(2*_y1),
                     "y2": -0.2/(2*_y1)}

        jac["g1"] = {"z1": -2*_y1*jac["y1"]["z1"],
                     "z2": -2*_y1*jac["y1"]["z2"],
                     "x1": -2*_y1*jac["y1"]["x1"],
                     "y2": -2*_y1*jac["y1"]["y2"]}

        return jac


class SellarDiscipline2(Discipline):
    def __init__(self, z1: Variable, z2: Variable,
                 y1: Variable, y2: Variable, g2: Variable):

        super().__init__(name="Disc2",
                         input_vars=[z1, z2, y1],
                         output_vars=[y2, g2])

    def _eval(self, inputs):
        _z1 = inputs["z1"]
        _z2 = inputs["z2"]
        _y1 = inputs["y1"]

        _y2 = np.abs(_y1) + _z1 + _z2

        return {"y2": _y2,
                "g2": _y2 - 24}

    def _differentiate(self, inputs, outputs):
        _y1 = inputs["y1"]

        jac = {}
        jac["y2"] = {"y1": np.sign(_y1), "z1": 1.0, "z2": 1.0}
        jac["g2"] = {"y1": np.sign(_y1), "z1": 1.0, "z2": 1.0}

        return jac


class SellarObjective(Discipline):
    def __init__(self, x1: Variable, z2: Variable,
                 y1: Variable, y2: Variable, f: Variable):

        super().__init__("Objective", [x1, z2, y1, y2], [f])

    def _eval(self, inputs):
        # Get the input variable values
        _x1 = inputs["x1"]
        _z2 = inputs["z2"]
        _y1 = inputs["y1"]
        _y2 = inputs["y2"]

        return {"f": _x1**2 + _z2 + _y1**2 + np.exp(-_y2)}

    def _differentiate(self, inputs, outputs):
        _x1 = inputs["x1"]
        _y1 = inputs["y1"]
        _y2 = inputs["y2"]

        jac = {"f": {}}
        jac["f"]["x1"] = 2*_x1
        jac["f"]["z2"] = 1.0
        jac["f"]["y1"] = 2 * _y1
        jac["f"]["y2"] = -np.exp(-_y2)

        return jac


# Design variables
z1 = Variable("z1", lb=-10, ub=10)
z2 = Variable("z2", lb=0, ub=10)
x1 = Variable("x1", lb=0, ub=10)

# Couplings
y1 = Variable("y1", lb=-100, ub=100)
y2 = Variable("y2", lb=-100, ub=100)

# Constraints and objective
g1 = Variable("g1", lb=-np.inf, ub=0, keep_feasible=False)
g2 = Variable("g2", lb=-np.inf, ub=0, keep_feasible=False)
f = Variable("f")

# Starting values
starting_values = {"x1": 1.0, "z1": 4, "z2": 3, "y1": 0.8, "y2": 0.9}

# Disciplines
disciplines = []
disciplines.append(SellarDiscipline1(z1, z2, x1, y2, y1, g1))
disciplines.append(SellarDiscipline2(z1, z2, y1, y2, g2))
disciplines.append(SellarObjective(x1, z2, y1, y2, f))
for disc in disciplines:
    disc.add_default_inputs(starting_values)

# Create MDA solver
solver = create_solver(disciplines,
                       n_iter_max=5,
                       type=SolverType.NONLINEAR_GS)

# Formulate using MDF
mdf = create_opt_problem("mdf",
                         disciplines,
                         [x1, z1, z2],
                         f, [g1, g2],
                         name="SellarMDF",
                         solver=solver)

# Create driver
mdf.driver = create_driver(mdf,
                           DriverType.SCIPY_DRIVER,
                           10,
                           1e-5)


# Solve the problem and plot objective's history
mdf.solve(starting_values)
mdf.plot_objective_history()
