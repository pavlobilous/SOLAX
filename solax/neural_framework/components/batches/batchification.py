import numpy as np
import jax
from jax.typing import ArrayLike

from dataclasses import dataclass, KW_ONLY
from collections import deque
from functools import wraps
from collections.abc import Callable
from typing import Generator

from .index_shuffle import *
from ....utils.multi_batching import *


CallableWithIndex = Callable[[int, ...], ...]        
        

@dataclass
class Inds:
    _: KW_ONLY
    key: ArrayLike = None
    length: bool
    
    def __post_init__(self):
        if self.key is not None:
            self.shuffled_inds = shuffled_inds(self.key, length=self.length)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, s):
        if self.key is not None:
            return self.shuffled_inds[s]
        else:
            return s
    

def batchify(*,
             batch_sz: int | None,
             shuffle: bool,
             num_args: int | None = None,
             multiple_devices: bool = False):
    """
    Batchify function "func" along the first "num_args" positional arguments (or all of them if =None).
    The batchified function is a generator function yielding outputs of "func" on batches.
    The zero-th argument of "func" is reserved for the batch index. See the type hints.
    All batchified arguments must have the same length (this is not asserted!).
    If "shuffle" is True, the data are shuffled before the generator starts yielding.
        --> In the latter case jax "key" needs to be passed as the first argument each time
            "batchified" function is called.
    If "multiple_devices" is True, up to "n_devices" batches are stacked together
        along a new dimension (which becomes axis=0);
        here "n_devices" is the number of available devices obtained automatically.
        --> In this case, the batched function is called on batches with an extra dimension.
    """
    def decorator(func: CallableWithIndex) -> Generator:
         
        @wraps(func)
        def batchified(*args, **kwargs):
            if shuffle:
                key, *args = args
            
            num_args_actual = num_args if (num_args is not None) else len(args)
            args_to_batch, args = args[:num_args_actual], args[num_args_actual:]
            try:
                arr_sz = len(args_to_batch[0])
            except IndexError:
                raise ValueError("No arguments to batch along.")
            b_sz = batch_sz or arr_sz
            
            inds = Inds(key=(key if shuffle else None), length=arr_sz)
            
            n_devices = jax.local_device_count() if multiple_devices else 1
            pack_edges = gen_pack_edges(arr_sz, b_sz, n_devices)
            for i, (b_start, b_end, num_pbs) in enumerate(pack_edges):
                args_batched = []
                for atb in args_to_batch:
                    ab = atb[inds[b_start:b_end]]
                    if multiple_devices:
                        ab = ab.reshape(num_pbs, -1, *ab.shape[1:])
                    args_batched.append(ab)
                yield func(i, *args_batched, *args, **kwargs)
                
        return batchified
    
    return decorator


def exhaust_batches(batchified_func: Callable):
    """
    Automatically exhaust generators created by "batchified_func" without memory overhead.
    """
    
    @wraps(batchified_func)
    def exh_func(*args, **kwargs):
        gen = batchified_func(*args, **kwargs)
        deque(gen, maxlen=0)
        
    return exh_func