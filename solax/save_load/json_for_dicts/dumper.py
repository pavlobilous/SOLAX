"""
Low-level (de)serialization of a nested dict that may contain NumPy
arrays anywhere inside it, to/from a directory on disk: plain values
are JSON-encoded into a single "schema.json", while each NumPy array is
written to its own ".npy" file at a path mirroring the array's key
hierarchy in the dict (e.g. dict_with_nd["a"]["b"] -> "<root>/a/b.npy").
This module has no notion of solax classes -- see
solax.save_load.dictification for turning registered solax objects into
plain nested dicts first.
"""
import json
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
import os
import shutil


@dataclass
class NDArrayWithPath:
    """
    A NumPy array paired with the on-disk location it should be saved
    to: "root_path" is the save's root directory and "rel_path" is the
    array's ".npy" path relative to it (mirroring its key hierarchy in
    the nested dict). Used internally by pack_keypaths_valarrs() to
    stand in for each array so JSONEncoderWithND can write it to disk
    as a side effect of JSON-encoding the surrounding dict.

    "is_scalar" records whether "arr" is itself a bare NumPy scalar
    (an np.generic instance, e.g. np.float64(0.5)) rather than a true
    ndarray. np.save/np.load do not preserve this distinction -- a
    scalar is written to ".npy" as, and loaded back as, a 0-d ndarray
    -- so it is recorded here at save time (see JSONEncoderWithND) and
    used by loader.py to reconstruct the original scalar on load.
    """
    root_path: str
    rel_path: str
    arr: NDArray[np.generic]
    is_scalar: bool


def save_arr(ndarr_wpath: NDArrayWithPath):
    """
    Writes "ndarr_wpath.arr" to disk as a ".npy" file at
    "ndarr_wpath.root_path"/"ndarr_wpath.rel_path", creating any missing
    intermediate directories first.
    """
    path = os.path.join(ndarr_wpath.root_path, ndarr_wpath.rel_path)
    directory, file = os.path.split(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(path, "bw") as f:
        np.save(f, ndarr_wpath.arr)


def pack_keypaths_valarrs(nested_dict: dict, root_path: str, rel_node_path: str) -> dict:
    """
    Recursively walks "nested_dict" and replaces every NumPy array
    value with an NDArrayWithPath recording where it should be saved:
    the relative path is built by joining "rel_node_path" with the
    array's key (plus ".npy"), so the array's location on disk mirrors
    its position in the nested dict. Nested dicts are recursed into
    (extending "rel_node_path" with the nested key); all other values
    are copied through unchanged. The array files themselves are not
    written here -- that happens lazily, as a side effect of JSON-encoding
    the resulting structure with JSONEncoderWithND (see dump_dict_with_nd()).

    Input:
        - "nested_dict": the dict to walk (must be a dict; raises
            TypeError otherwise -- this is also what save() surfaces
            when e.g. a bare top-level NumPy array is passed to it).
        - "root_path": the save's root directory (stored in each
            NDArrayWithPath, unused during the walk itself).
        - "rel_node_path": the key path accumulated so far, relative to
            "root_path" (start with "" at the top level).
    Output:
        A new nested dict of the same shape as "nested_dict", with
        NumPy arrays replaced by NDArrayWithPath instances.
    """
    if not isinstance(nested_dict, dict):
        raise TypeError('"nested_dict" must be a dict.')
    res_nested_dict = {}
    for k, v in nested_dict.items():
        if isinstance(v, np.ndarray | np.generic):
            rel_path = os.path.join(rel_node_path, k + ".npy")
            res_nested_dict[k] = NDArrayWithPath(root_path, rel_path, v, isinstance(v, np.generic))
        elif isinstance(v, dict):
            rel_path = os.path.join(rel_node_path, k)
            res_nested_dict[k] = pack_keypaths_valarrs(v, root_path, rel_path)
        else:
            res_nested_dict[k] = v
    return res_nested_dict


class JSONEncoderWithND(json.JSONEncoder):
    """
    A json.JSONEncoder that additionally knows how to handle
    NDArrayWithPath instances (as produced by pack_keypaths_valarrs()):
    each one is written to its own ".npy" file as a side effect of
    encoding (via save_arr()), and is represented in the resulting JSON
    only by a small placeholder dict, {".ndarray_path": rel_path,
    ".is_scalar": ...}, which "loader.py"'s object_hook uses to load the
    array back in (".is_scalar" telling it whether to hand back a bare
    NumPy scalar or a true ndarray, see NDArrayWithPath).

    A bare NumPy array/scalar reaching this encoder (i. e. one that was
    not first wrapped in an NDArrayWithPath by pack_keypaths_valarrs())
    is a usage error: it has no associated save path, so it raises
    TypeError instead of being encoded.
    """

    def default(self, arg):
        """Handles NDArrayWithPath/bare-array values; see class docstring."""
        if isinstance(arg, np.ndarray | np.generic):
            raise TypeError("Standalone NumPy objects are not serializable and not savable. Associated path needed.")
        elif isinstance(arg, NDArrayWithPath):
            save_arr(arg)
            return {".ndarray_path" : arg.rel_path, ".is_scalar": arg.is_scalar}
        else:
            return super().default(arg)


def dump_dict_with_nd(dict_with_nd: dict, path: str):
    """
    Serializes "dict_with_nd" which is a nested dictionary
        with occasional NumPy arrays.
    Each NumPy array is saved at the path
        constructed from the key hyerarchy with prepended root "path".

    Warning:
        "path" is treated as a directory: if it already exists it is
        first removed entirely (shutil.rmtree) before being recreated,
        so any previous contents at "path" are erased before saving.

    Writes a "schema.json" directly under "path" holding the
    JSON-encoded structure (with each array replaced by a
    {".ndarray_path": ..., ".is_scalar": ...} placeholder, see
    JSONEncoderWithND), plus one ".npy" file per NumPy array found
    anywhere in "dict_with_nd", each at a path mirroring its key
    hierarchy.

    Raises:
        TypeError: if "dict_with_nd" is not itself a dict (see
            pack_keypaths_valarrs()).
    """
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    packed_dict = pack_keypaths_valarrs(dict_with_nd, path, "")
    with open(os.path.join(path, "schema.json"), "w") as f:
        json.dump(packed_dict, f, cls=JSONEncoderWithND)