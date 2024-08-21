import os
import json
import numpy as np


def get_object_hook(root_path: str):

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
    """
    object_hook = get_object_hook(path)
    with open(os.path.join(path, "schema.json"), "r") as f:
        dict_with_nd = json.load(f, object_hook=object_hook)
    return dict_with_nd