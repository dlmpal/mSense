from typing import Dict
from os.path import exists, splitext

from numpy import ndarray
import h5py

from msense.core.constants import FLOAT_DTYPE
from msense.utils.array_and_dict_utils import check_values_match
from msense.cache.cache import Cache, CachePolicy


class HDF5Cache(Cache):
    def __init__(self, **kwargs) -> None:
        # Entries is a hdf5 file
        self.entries: h5py.File = None
        # Add .hdf5 suffix to cache path
        _, suffix = splitext(kwargs["path"])
        if suffix != ".hdf5":
            kwargs["path"] += ".hdf5"
        super().__init__(**kwargs)

    def check_if_entry_exists(self, input_values: Dict[str, ndarray]):
        if not self.entries:
            return None
        for entry_idx in reversed(self.entries):
            if check_values_match(self.input_vars,
                                  self.entries[entry_idx]["inputs"],
                                  input_values,
                                  self.tol):
                return self.entries[entry_idx]
        return None

    def add_entry(self, input_values: Dict[str, ndarray], output_values: Dict[str, ndarray] = None,
                  jac: Dict[str, Dict[str, ndarray]] = None) -> None:
        new_entry = None
        if self.policy == CachePolicy.LATEST:
            if self.entries:
                del self.entries["0"]
        else:
            new_entry = self.check_if_entry_exists(input_values)

        # If entry does not exist, instantiate new one
        if new_entry is None:
            new_entry = self.entries.create_group(str(len(self.entries)))
            new_entry.create_group("inputs")
            new_entry.create_group("outputs")
            new_entry.create_group("jac")
            # Add inputs
            for var in self.input_vars:
                new_entry["inputs"].create_dataset(
                    var.name, data=input_values[var.name], dtype=FLOAT_DTYPE)

        # Modify entry outputs, if given
        if output_values is not None:
            for var in self.output_vars:
                new_entry["outputs"].require_dataset(
                    name=var.name, data=output_values[var.name], shape=(var.size), dtype=FLOAT_DTYPE)

        # Modify entry jac, if given
        if jac is not None:
            for out_var in self.doutput_vars:
                new_entry["jac"].require_group(out_var.name)
                for in_var in self.dinput_vars:
                    new_entry["jac"][out_var.name].require_dataset(name=in_var.name, data=jac[out_var.name][in_var.name],
                                                                   shape=(out_var.size, in_var.size), dtype=FLOAT_DTYPE)

    def load_entry(self, input_values: Dict[str, ndarray]):
        entry = self.check_if_entry_exists(input_values)
        if entry is not None:
            outputs = None
            if entry["outputs"]:
                outputs = {var.name: entry["outputs"]
                           [var.name][()].copy() for var in self.output_vars}
            jac = None
            if len(entry["jac"]) > 0:
                jac = {out_var.name: {in_var.name: entry["jac"][out_var.name][in_var.name][()].copy() for in_var in self.dinput_vars}
                       for out_var in self.doutput_vars}
            return outputs, jac
        else:
            return None, None

    def from_disk(self):
        # Create the hdf5 file,
        # if it doesn't exist
        if not exists(self.path):
            self.entries = h5py.File(self.path, "w")
        self.entries = h5py.File(self.path, "r+")

    def to_disk(self, path: str = None):
        # Already stored on disk
        pass
