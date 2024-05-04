from typing import List
from enum import Enum

from msense.core.variable import Variable
from msense.cache.cache import Cache, CachePolicy
from msense.cache.memory_cache import MemoryCache
from msense.cache.hdf5_cache import HDF5Cache


class CacheType(str, Enum):
    MEMORY = "memory"
    HDF5 = "hdf5"


def create_cache(input_vars: List[Variable], output_vars: List[Variable],
                 dinput_vars: List[Variable], doutput_vars: List[Variable],
                 type: CacheType = CacheType.MEMORY, policy: CachePolicy = CachePolicy.LATEST,
                 tol: float = 1e-9, path: str = None) -> Cache:
    kwargs = {"path": path, "input_vars": input_vars, "output_vars": output_vars,
              "dinput_vars": dinput_vars, "doutput_vars": doutput_vars,
              "policy": policy, "tol": tol}
    if type is None:
        return None
    if type == CacheType.MEMORY:
        return MemoryCache(**kwargs)
    if type == CacheType.HDF5:
        return HDF5Cache(**kwargs)
    else:
        raise ValueError(f"CacheType {type} is not available.")
