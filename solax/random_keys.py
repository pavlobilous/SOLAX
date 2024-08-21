import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from collections.abc import Iterator

from .save_load import *


def init_from_key(_key):
    rk = RandomKeys(seed=0)
    rk._key = _key
    return rk


@dataclass
class RandomKeys(Iterator):
    _key: jnp.array
    
    
    @classmethod
    def fake_key(cls):
        return jax.random.key(0)
    
    
    def __init__(self, *, seed: int):
        self._key = jax.random.PRNGKey(seed)
        
    
    def __iter__(self):
        return self
    
    
    def __next__(self):
        self._key, subkey = jax.random.split(self._key)
        return subkey
    
    
    def __pre_dictify__(self):
        return init_from_key(np.array(self._key))
    
    
    def __post_undictify__(self):
        return init_from_key(jnp.array(self._key))
    
    
    
save_load_registry.register("RandomKeys", RandomKeys, init_from_key)