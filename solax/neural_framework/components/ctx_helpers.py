from functools import wraps
from contextlib import nullcontext


def null_if_arg0_none(ctx_gen):
    
    @wraps(ctx_gen)
    def ctx_gen_wnull(*args, **kwargs):
        ctx = ctx_gen(*args, **kwargs) if (args[0] is not None) else nullcontext()
        return ctx
        
    return ctx_gen_wnull