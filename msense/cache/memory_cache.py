from typing import Dict
from os.path import exists, splitext
from logging import getLogger

from numpy import ndarray
import json

from msense.utils.array_and_dict_utils import check_values_match
from msense.utils.array_and_dict_utils import copy_dict_1d, copy_dict_2d
from msense.utils.array_and_dict_utils import verify_dict_1d, verify_dict_2d
from msense.cache.cache import Cache, CachePolicy

logger = getLogger(__name__)


class MemoryCache(Cache):
    """
    A cache stored entirely in memory. 
    Can be saved as a json file, and loaded at runtime.
    """

    def __init__(self, **kwargs) -> None:
        # Ensure that the filename has a .json extension
        if kwargs["path"] is not None:
            path, ext = splitext(kwargs["path"])
            if ext != ".json":
                kwargs["path"] = path + ".json"

        # Entries is a list of dictionaries
        # The last item of the list is the latest
        # entry in the cache
        self._entries = []

        super().__init__(**kwargs)

    def check_if_entry_exists(self, input_values: Dict[str, ndarray]):
        for entry in reversed(self._entries):
            if check_values_match(self.input_vars, entry["inputs"],
                                  input_values, self.tol):
                return entry
        return None

    def add_entry(self, input_values: Dict[str, ndarray],
                  output_values: Dict[str, ndarray] = None,
                  jac: Dict[str, Dict[str, ndarray]] = None) -> None:
        # Check if an entry exists for the given inputs
        entry = self.check_if_entry_exists(input_values)

        # If the entry does not exist, create a new one
        entry_exists = True
        if entry is None:
            entry = {"inputs": copy_dict_1d(self.input_vars, input_values),
                     "outputs": {},
                     "jac": {}}
            entry_exists = False

        if output_values is not None:
            entry["outputs"] = copy_dict_1d(self.output_vars, output_values)

        if jac is not None:
            entry["jac"] = copy_dict_2d(
                self.dinput_vars, self.doutput_vars, jac)

        if self.policy == CachePolicy.FULL and entry_exists is False:
            self._entries.append(entry)

        if self.policy == CachePolicy.LATEST:
            self._entries = [entry]

    def load_entry(self, input_values: Dict[str, ndarray]):
        entry = self.check_if_entry_exists(input_values)
        if entry is not None:
            return copy_dict_1d(self.output_vars, entry["outputs"]), \
                copy_dict_2d(self.dinput_vars, self.doutput_vars, entry["jac"])
        else:
            return None, None

    def from_file(self):
        if not exists(self.path):
            logger.warn(
                f"MemoryCache cannot be loaded from file. File: {self.path} does not exist.")
            return

        with open(self.path, "r") as file:
            # Load the json object
            json_obj = json.load(file)

            self._entries = []
            for _, entry in json_obj.items():
                # Load input values
                entry["inputs"] = verify_dict_1d(
                    self.input_vars, entry["inputs"])

                # Load output values, if they exist
                if entry["outputs"]:
                    entry["outputs"] = verify_dict_1d(
                        self.output_vars, entry["outputs"])

                # Load jacobian, if it exists
                if entry["jac"]:
                    entry["jac"] = verify_dict_2d(
                        self.dinput_vars, self.doutput_vars, entry["jac"])

                # Add the entry
                self._entries.append(entry)

        if self.policy == CachePolicy.LATEST:
            self._entries = [self._entries[-1]]

    def to_file(self):
        # Convert the list of entries to dictionary
        entry_dict = {}
        for entry_idx, entry in enumerate(self._entries):
            # Create a new entry in the json dictionary
            entry_dict[entry_idx] = {"inputs": {}, "outputs": {}, "jac": {}}

            # Write input values
            for in_var in self.input_vars:
                entry_dict[entry_idx]["inputs"][in_var.name] = entry["inputs"][in_var.name].tolist(
                )

            # Write output, if they exist
            if entry["outputs"]:
                for out_var in self.output_vars:
                    entry_dict[entry_idx]["outputs"][out_var.name] = entry["outputs"][out_var.name].tolist(
                    )

            # Write jacobian, if it exists
            if entry["jac"]:
                for out_var in self.doutput_vars:
                    entry_dict[entry_idx]["jac"][out_var.name] = {}
                    for in_var in self.dinput_vars:
                        entry_dict[entry_idx]["jac"][out_var.name][in_var.name] = entry["jac"][out_var.name][in_var.name].tolist(
                        )

        # Create a json object from the entry dictionary
        # and save the object to file
        json_obj = json.dumps(entry_dict, indent=4)
        with open(self.path, "w") as file:
            file.write(json_obj)
