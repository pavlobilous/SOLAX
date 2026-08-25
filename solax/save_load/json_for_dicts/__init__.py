"""
Low-level, class-agnostic (de)serialization of nested Python dicts that
may contain NumPy arrays anywhere inside them: plain values go through
the standard "json" module, while each array is saved/loaded via
NumPy's own array serialization at a path mirroring its position in the
dict. Used by solax.save_load.usr_save_load as the on-disk half of
save()/load(), after solax.save_load.dictification has turned any
registered SOLAX objects into plain dicts.
"""
from .dumper import dump_dict_with_nd
from .loader import load_dict_with_nd