from typing import TypeVar

from .dictification import *
from .json_for_dicts import *


SolaxClass = TypeVar("SolaxClass")

def save(arg: SolaxClass | dict, path: str):
    """
    This function is a tool for saving all necessary data at once.
    
    It can save:
        -> a standalone SOLAX object
        -> a nested dictionary with SOLAX objects and NumPy arrays inside.
        
    Note:
        Since SOLAX uses dictionary keys as parts of saving paths,
        all keys in the saved dictionary must be:
            (1) of string type;
            (2) valid variable identifiers,
                i. e. each "key" could potentially be a Python variable name.
        It is recommended to create these dicts using the "dict" constructor:
            dict(key1=value1, key2=value2, ...)
        In this case Python won't allow the keys to have a wrong format.
        
    Examples:
        pass
    """
    try:
        dct = dictify(arg)
        dump_dict_with_nd(dct, path)
    except TypeError as e:
        raise TypeError('Something wrong passed to the saver. See help(save).')


def load(path: str) -> SolaxClass | dict:
    """
    Loads the data saved previously using the "save" function. See help(save).
    """
    return undictify(load_dict_with_nd(path))