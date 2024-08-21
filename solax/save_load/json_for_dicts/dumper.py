import json
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
import os, shutil


@dataclass
class NDArrayWithPath:
    root_path: str
    rel_path: str
    arr: NDArray[np.generic]
    

def save_arr(ndarr_wpath: NDArrayWithPath):
    path = os.path.join(ndarr_wpath.root_path, ndarr_wpath.rel_path)
    directory, file = os.path.split(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(path, "bw") as f:
        np.save(f, ndarr_wpath.arr)
        

def pack_keypaths_valarrs(nested_dict: dict, root_path: str, rel_node_path: str) -> dict:
    if not isinstance(nested_dict, dict):
        raise TypeError('"nested_dict" must be a dict.')
    res_nested_dict = {}
    for k, v in nested_dict.items():
        if isinstance(v, np.ndarray | np.generic):
            rel_path = os.path.join(rel_node_path, k + ".npy")
            res_nested_dict[k] = NDArrayWithPath(root_path, rel_path, v)
        elif isinstance(v, dict):
            rel_path = os.path.join(rel_node_path, k)
            res_nested_dict[k] = pack_keypaths_valarrs(v, root_path, rel_path)
        else:
            res_nested_dict[k] = v
    return res_nested_dict
    

class JSONEncoderWithND(json.JSONEncoder):
    
    def default(self, arg):
        if isinstance(arg, np.ndarray | np.generic):
            raise TypeError("Standalone NumPy objects are not serializable and not savable. Associated path needed.")
        elif isinstance(arg, NDArrayWithPath):
            save_arr(arg)
            return {".ndarray_path" : arg.rel_path}
        else:
            return super().default(arg)
            

def dump_dict_with_nd(dict_with_nd: dict, path: str):
    """
    Serializes "dict_with_nd" which is a nested dictionary
        with occasional NumPy arrays.
    Each NumPy array is saved at the path
        constructed from the key hyerarchy with prepended root "path".
    """
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    packed_dict = pack_keypaths_valarrs(dict_with_nd, path, "")
    with open(os.path.join(path, "schema.json"), "w") as f:
        json.dump(packed_dict, f, cls=JSONEncoderWithND)