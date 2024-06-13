from typing import List
from enum import Enum

from msense.core.variable import Variable
from msense.cache.cache import Cache, CachePolicy
from msense.cache.memory_cache import MemoryCache


class CacheType(str, Enum):
    MEMORY = "memory"


def create_cache(input_vars: List[Variable],
                 output_vars: List[Variable],
                 dinput_vars: List[Variable],
                 doutput_vars: List[Variable],
                 type: CacheType = CacheType.MEMORY,
                 policy: CachePolicy = CachePolicy.LATEST,
                 tol: float = 1e-9,
                 path: str = None) -> Cache:

    kwargs = {"input_vars": input_vars,
              "output_vars": output_vars,
              "dinput_vars": dinput_vars,
              "doutput_vars": doutput_vars,
              "policy": CachePolicy(policy),
              "tol": tol,
              "path": path}

    if type is None:
        return None

    type = CacheType(type)

    if type == CacheType.MEMORY:
        return MemoryCache(**kwargs)
