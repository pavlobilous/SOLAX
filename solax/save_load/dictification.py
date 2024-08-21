from .registration import save_load_registry as slr


def assert_valid_key(k):
    if not isinstance(k, str):
        raise TypeError('All dict keys must be strings.')
    if not k.isidentifier():
        raise ValueError('All dict keys must be valid variable identifiers.')


def dictify(arg):
    """
    Translates "arg" to a Python dict.
    
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
        arg_atts = dictify(getattr(arg, "__dict__"))
    
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
    Translates a Python dict "args" to what it was before dictification.
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