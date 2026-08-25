"""
The counterpart to "dumper.py": reconstructs a nested dict (with NumPy
arrays restored in place) from a directory previously written by
dump_dict_with_nd() -- a "schema.json" plus one ".npy" file per array.
"""
import os
import json
import numpy as np


def get_object_hook(root_path: str):
    """
    Builds a json.load "object_hook" bound to "root_path" (the save's
    root directory).

    The returned "object_hook" is called by the json module on every
    JSON object as it is decoded, innermost first. If the object is one
    of dumper.py's array placeholders, {".ndarray_path": rel_path}, it
    is replaced by the actual array loaded from
    "root_path"/"rel_path" (written by dumper.py's save_arr()); any
    other dict is passed through unchanged (leaving further
    interpretation, e.g. of ".class"-tagged dicts, to undictify()).
    """

    def object_hook(d):
        rel_path = d.get(".ndarray_path")
        if rel_path is not None:
            path = os.path.join(root_path, rel_path)
            with open(path, "rb") as f:
                arr = np.load(f)
            return arr
        else:
            return d

    return object_hook



def load_dict_with_nd(path: str) -> dict:
    """
    Deserializes the nested dict saved at "path" with loading NumPy arrays.

    Reads "path"/"schema.json" (written by dump_dict_with_nd()) and,
    while decoding it, replaces each ".ndarray_path" placeholder with
    the corresponding array loaded from disk (see get_object_hook()),
    yielding a nested dict structurally identical to the one originally
    passed to dump_dict_with_nd().
    """
    object_hook = get_object_hook(path)
    with open(os.path.join(path, "schema.json"), "r") as f:
        dict_with_nd = json.load(f, object_hook=object_hook)
    return dict_with_nd