"""
The user-facing entry point of solax's save/load subsystem:
"save"/"load" combine dictification (solax.save_load.dictification --
turning registered solax objects into plain nested dicts) with
solax.save_load.json_for_dicts (turning those dicts, and any NumPy
arrays nested inside them, into files on disk and back).

"pickle" is deliberately not used anywhere in this subsystem: as the
solax paper notes, it has been shown to have safety flaws (unpickling
untrusted data can execute arbitrary code), so solax instead uses the
standard "json" module for plain values plus NumPy's own array
(de)serialization for arrays, with classes opting in via an explicit
registry (see solax.save_load.registration).
"""
from typing import TypeVar

from .dictification import *
from .json_for_dicts import *


SolaxClass = TypeVar("SolaxClass")

def save(arg: SolaxClass | dict, path: str):
    """
    This function is a tool for saving all necessary data at once.

    It can save:
        -> a standalone solax object
        -> a nested dictionary with solax objects and NumPy arrays inside.

    Warning:
        "path" is treated as a directory: it is created, and -- if it
        already exists -- first ERASED (via shutil.rmtree, see
        dump_dict_with_nd()) and recreated from scratch. Saving to a
        path that already holds something you care about will silently
        discard it.

    Note:
        Since solax uses dictionary keys as parts of saving paths,
        all keys in the saved dictionary must be:

            (1) of string type;
            (2) valid variable identifiers,
                i. e. each "key" could potentially be a Python variable name.

        It is recommended to create these dicts using the "dict" constructor:

            dict(key1=value1, key2=value2, ...)

        In this case Python won't allow the keys to have a wrong format.

    Raises:
        TypeError: if "arg" cannot be dictified -- e.g. it is a bare
            value that is neither a registered solax object nor a dict
            (a standalone NumPy array included), it contains an instance
            of an unregistered class, or one of its dict keys (own or
            nested) is not a string. All such failures from dictify()
            are caught here and re-raised as this same generic TypeError.
        ValueError: if one of "arg"'s dict keys (own or nested) is a
            string but not a valid Python identifier (see
            assert_valid_key() in solax.save_load.dictification) --
            unlike the TypeError cases above, this exception is *not*
            caught here and propagates from dictify() unchanged, with
            its own more specific message.

    Examples:
        pass
    """
    try:
        dct = dictify(arg)
        dump_dict_with_nd(dct, path)
    except TypeError as e:
        raise TypeError('Something wrong passed to the saver. See help(save).') from e


def load(path: str) -> SolaxClass | dict:
    """
    Loads the data saved previously using the "save" function. See help(save).

    Reads back the on-disk structure written by save() (a "schema.json"
    plus any ".npy" array files under "path") and undictifies it into
    the original solax object or nested dict.
    """
    return undictify(load_dict_with_nd(path))