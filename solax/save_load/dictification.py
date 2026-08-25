"""
Translation of registered solax objects (and plain nested dicts mixing
them with NumPy arrays and standard Python types) to/from plain nested
Python dicts -- the layer between solax's own classes and the
JSON/NumPy-based (de)serialization in solax.save_load.json_for_dicts.

"dictify"/"undictify" only ever produce/consume plain dicts (and the
Python/NumPy values nested inside them); nothing is written to disk
here -- see solax.save_load.usr_save_load.save/load for that. Classes
must be registered beforehand via "save_load_registry"
(solax.save_load.registration) to be dictifiable at all.
"""
from .registration import save_load_registry as slr


def assert_valid_key(k):
    """
    Checks that "k" is usable as a dict key in a dictified structure.

    solax uses dict keys both for JSON encoding and (when a NumPy array
    is found under that key) as part of the on-disk saving path, so a
    key must be a string that is also a valid Python identifier (i. e.
    it could be used as a Python variable name).

    Raises:
        TypeError: if "k" is not a string.
        ValueError: if "k" is a string but not a valid identifier
            (e.g. it starts with a digit, or contains a space).
    """
    if not isinstance(k, str):
        raise TypeError('All dict keys must be strings.')
    if not k.isidentifier():
        raise ValueError('All dict keys must be valid variable identifiers.')


def dictify(arg):
    """
    Translates "arg" to a Python dict (recursively).

    "arg" may be:
        - a plain value (int, str, NumPy array, ...), returned unchanged;
        - a dict, whose keys are checked with assert_valid_key() and
            whose values are dictified recursively;
        - an instance of a class registered in "save_load_registry"
            (detected via hasattr(arg, "__dict__")), which is turned into
            {".class": label, ".attrs": dictify(vars(arg))} -- where
            "label" is the class's registered label. If "arg" also
            behaves like a dict (e.g. subclasses dict), its own dict
            items are additionally stored under ".dict". If "arg" has a
            "__pre_dictify__" method, it is called first and its return
            value (rather than "arg" itself) is dictified -- this lets a
            class present a JSON/dict-friendly stand-in for itself (e.g.
            to remap non-string dict keys, or convert framework-specific
            data to plain NumPy).
        - an instance of a registered class that defines "__save__"
            (opting out of dictification), which is reduced to just
            {".class_with_own_svld": label} -- the class is expected to
            handle its own saving/loading and undictify() will hand back
            the class itself, not a reconstructed instance.

    Raises:
        TypeError: if "arg" has an unregistered class, or a dict (own or
            nested) has a non-string key (see assert_valid_key()).
        ValueError: if a dict (own or nested) has a string key that is
            not a valid Python identifier (see assert_valid_key()).

    Note 1:
        Classes must be priorly registered in "save_load_registry" to be dicitifiable.

    Note 2:
        This is sometimes called serialization (as e.g. in the "serpy" library).
        But also saving / loading is called serialization.
        So we use more explicite term "dictification" to avoid confusions.
        Nothing is saved here, but only translated to dicts.
    """
    arg_is_cls = hasattr(arg, "__dict__")
    arg_is_dict = isinstance(arg, dict)

    if arg_is_cls:
        label = slr.retreive_label(arg.__class__)
        if hasattr(arg, "__save__"):
            return {".class_with_own_svld": label}
        if hasattr(arg, "__pre_dictify__"):
            arg = arg.__pre_dictify__()
        arg_atts = dictify(arg.__dict__)
    
    if arg_is_dict:
        arg_dict = {}
        for k, v in arg.items():
            assert_valid_key(k)
            arg_dict[k] = dictify(v)

    if arg_is_cls:
        arg = {
            ".class": label,
            ".attrs": arg_atts,
        }
        if arg_is_dict:
            arg[".dict"] = arg_dict
    elif arg_is_dict:
        arg = arg_dict
            
    return arg


def undictify(arg):
    """
    Translates a Python dict "arg" back to what it was before
    dictification (the inverse of dictify()).

    Non-dict "arg" values are returned unchanged. A dict is interpreted
    based on its special keys:
        - ".class_with_own_svld": the registered class itself is
            returned (not an instance -- classes opting out of
            dictification via "__save__" are left for their own
            loading machinery to reconstruct).
        - ".class": the registered class's reconstruction callable is
            looked up and called with the undictified ".attrs" as
            keyword arguments; if ".dict" is also present (the class
            behaves like a dict), its undictified items are copied onto
            the new instance via "instance.update(...)". If the
            resulting instance has a "__post_undictify__" method, it is
            called and its return value used instead (the inverse of
            "__pre_dictify__").
        - otherwise: an ordinary (possibly nested) dict, whose values
            are undictified in place and whose (possibly modified) self
            is returned.
    """
    if not isinstance(arg, dict):
        return arg
    elif ".class_with_own_svld" in arg:
        cls = slr.retreive_cls(arg[".class_with_own_svld"])
        return cls
    elif ".class" in arg:
        init_from_attr = slr.retreive_init(arg[".class"])
        attrs = undictify(arg[".attrs"])
        instance = init_from_attr(**attrs)
        if ".dict" in arg:
            instance.update({k: undictify(v) for k, v in arg[".dict"].items()})
        if hasattr(instance, "__post_undictify__"):
            instance = instance.__post_undictify__()
        return instance
    else:
        for k, v in arg.items():
            arg[k] = undictify(v)
        return arg