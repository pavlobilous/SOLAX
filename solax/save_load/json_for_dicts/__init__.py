"""
Low-level, class-agnostic (de)serialization of nested Python dicts that
may contain NumPy arrays anywhere inside them: plain values go through
the standard "json" module, while each array is saved/loaded via
NumPy's own array serialization at a path mirroring its position in the
dict.
"""
from .dumper import dump_dict_with_nd
from .loader import load_dict_with_nd