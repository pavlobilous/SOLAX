"""
Generic decorator for turning a context-manager-producing function
into a no-op (nullcontext) whenever its first positional argument is
None, so that callers can pass an "optional" object (e.g. an absent
monitoring/logging object) through "with" blocks without special-casing
None themselves.
"""
from functools import wraps
from contextlib import nullcontext


def null_if_arg0_none(ctx_gen):
    """
    Decorator for a context-manager-generator function "ctx_gen" (i.e.
    a function decorated with @contextmanager). Returns a wrapped
    version that, when called, produces "ctx_gen"'s context manager as
    usual if its first positional argument is not None, or an inert
    contextlib.nullcontext() if that argument is None -- so a function
    taking an optional "ctx"-like first argument can be entered with
    None and simply do nothing.
    """

    @wraps(ctx_gen)
    def ctx_gen_wnull(*args, **kwargs):
        ctx = ctx_gen(*args, **kwargs) if (args[0] is not None) else nullcontext()
        return ctx

    return ctx_gen_wnull