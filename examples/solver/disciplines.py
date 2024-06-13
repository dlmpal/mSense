import numpy as np
from msense.api import *


def func1(input_vars):
    x1 = input_vars["x1"][0]
    z = input_vars["z"][0]
    y12 = input_vars["y12"][0]
    y21 = x1**2 + x1*z - y12 * z
    return {"y21": y21}


def dfunc1(input_vars):
    x1 = input_vars["x1"][0]
    z = input_vars["z"][0]
    y12 = input_vars["y12"][0]

    dx1 = 2 * x1 + z
    dz = np.array([x1 - y12])
    dy12 = np.array([-z])
    jac = {}
    jac["y21"] = {"x1": dx1, "z": dz, "y12": dy12}

    return jac


def func2(input_vars):
    x2 = input_vars["x2"][0]
    z = input_vars["z"][0]
    y21 = input_vars["y21"][0]
    y12 = 2*y21 - x2**2 + z*x2
    return {"y12": np.array([y12])}


def dfunc2(input_vars):
    x2 = input_vars["x2"][0]
    z = input_vars["z"][0]
    y21 = input_vars["y21"][0]
    dx2 = np.array([-2*x2 + z])
    dz = np.array([x2])
    dy21 = np.array([2])
    jac = {}
    jac["y12"] = {"x2": dx2, "z": dz, "y21": dy21}
    return jac


class SimpleDisc(Discipline):
    def __init__(self, name, input_vars, output_vars, func, dfunc):
        self.func = func
        self.dfunc = dfunc
        super().__init__(name, input_vars, output_vars,
                         cache_policy=CachePolicy.FULL, cache_type=CacheType.MEMORY)

    def _eval(self) -> None:
        self._values.update(self.func(self._values))

    def _differentiate(self) -> None:
        self._jac.update(self.dfunc(self._values))
