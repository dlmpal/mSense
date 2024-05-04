from typing import Dict, List
from os.path import exists, splitext

from numpy import ndarray
import json

from msense.core.variable import Variable
from msense.utils.array_and_dict_utils import check_values_match
from msense.utils.array_and_dict_utils import copy_dict_1d, copy_dict_2d
from msense.utils.array_and_dict_utils import verify_dict_1d, verify_dict_2d
from msense.cache.cache import Cache, CachePolicy


class MemoryCache(Cache):
    def __init__(self, **kwargs) -> None:
        # Entries is a list of dictionaries
        self.entries = []
        # Add .json suffix to cache path
        _, suffix = splitext(kwargs["path"])
        if suffix != ".json":
            kwargs["path"] += ".json"
        super().__init__(**kwargs)

    def check_if_entry_exists(self, input_values: Dict[str, ndarray]):
        if not self.entries:
            return None
        for entry in reversed(self.entries):
            if check_values_match(self.input_vars,
                                  entry["inputs"],
                                  input_values,
                                  self.tol):
                return entry
        return None

    def add_entry(self,
                  input_values: Dict[str, ndarray],
                  output_values: Dict[str, ndarray] = None,
                  jac: Dict[str, Dict[str, ndarray]] = None) -> None:
        # If entry does not exist, instantiate new one
        new_entry = self.check_if_entry_exists(input_values)
        if new_entry is None:
            new_entry = {"inputs": copy_dict_1d(self.input_vars, input_values),
                         "outputs": {},
                         "jac": {}}
            entry_exists = False
        else:
            entry_exists = True
        # Modify entry outputs, if given
        if output_values is not None:
            new_entry["outputs"] = copy_dict_1d(
                self.output_vars, output_values)
        # Modify entry jac, if given
        if jac is not None:
            new_entry["jac"] = copy_dict_2d(
                self.dinput_vars, self.doutput_vars, jac)
        # Clear the entries and append, if required
        if self.policy == CachePolicy.LATEST:
            self.entries = []
        if entry_exists == False or len(self.entries) == 0:
            self.entries.append(new_entry)

    def load_entry(self, input_values: Dict[str, ndarray]):
        entry = self.check_if_entry_exists(input_values)
        if entry is not None:
            return copy_dict_1d(self.output_vars, entry["outputs"]), \
                copy_dict_2d(self.dinput_vars, self.doutput_vars, entry["jac"])
        else:
            return None, None

    def from_disk(self):
        if not exists(self.path):
            return
        with open(self.path, "r") as file:
            json_obj = json.load(file)
            self.entries = []
            for _, entry in json_obj.items():
                # Load inputs
                entry["inputs"] = verify_dict_1d(
                    self.input_vars, entry["inputs"])
                # Load outputs
                if entry["outputs"]:
                    entry["outputs"] = verify_dict_1d(
                        self.output_vars, entry["outputs"])
                # Load jacobian, if exists
                if entry["jac"]:
                    entry["jac"] = verify_dict_2d(
                        self.dinput_vars, self.doutput_vars, entry["jac"])
                # Add the entry
                self.entries.append(entry)

    def to_disk(self):
        json_obj = {}
        for entry_idx, entry in enumerate(self.entries):
            json_obj[entry_idx] = {"inputs": {}, "outputs": {}, "jac": {}}
            # Write inputs
            for in_var in self.input_vars:
                json_obj[entry_idx]["inputs"][in_var.name] = entry["inputs"][in_var.name].tolist(
                )
            # Write outputs
            if entry["outputs"]:
                for out_var in self.output_vars:
                    json_obj[entry_idx]["outputs"][out_var.name] = entry["outputs"][out_var.name].tolist(
                    )
            # Write jacobian, if exists
            if entry["jac"]:
                for dout_var in self.doutput_vars:
                    json_obj[entry_idx]["jac"][dout_var.name] = {}

                    for din_var in self.dinput_vars:
                        json_obj[entry_idx]["jac"][dout_var.name][din_var.name] = entry["jac"][dout_var.name][din_var.name].tolist(
                        )
        json_obj = json.dumps(json_obj, indent=4)
        with open(self.path, "w") as file:
            file.write(json_obj)
