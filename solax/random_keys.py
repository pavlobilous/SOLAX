"""
RandomKeys: a stateful iterator handing out fresh, reproducible JAX PRNG
subkeys via next(), wrapping jax.random.PRNGKey/split so callers don't
have to thread and split keys by hand.
"""
import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from collections.abc import Iterator

from .save_load import *


def init_from_key(_key):
    """Builds a RandomKeys directly from an already-existing key "_key"
    (JAX or NumPy), bypassing the seed-based __init__. Used internally
    and to reconstruct a RandomKeys when loading it back with
    solax.load()."""
    rk = RandomKeys(seed=0)
    rk._key = _key
    return rk


@dataclass
class RandomKeys(Iterator):
    """
    An Iterator over JAX PRNG subkeys: each next(rk) call splits off and
    returns a fresh subkey, advancing the internal state so consecutive
    calls never repeat a key. Construct with RandomKeys(seed=...) for a
    reproducible sequence seeded via jax.random.PRNGKey.

    RandomKeys has its own __pre_dictify__/__post_undictify__ hooks for
    solax.save()/solax.load(): it is saved in its current iteration
    state (the current internal key, as a plain NumPy array), rather
    than as any of the JAX keys it has generated -- so loading a saved
    RandomKeys resumes the same reproducible sequence from where it
    left off.
    """
    _key: jnp.array


    @classmethod
    def fake_key(cls):
        """
        Returns a fixed, non-advancing JAX PRNG key, for use as a
        throwaway initialization key where the actual initialized
        weights don't matter -- e.g. when reconstructing a
        BasisClassifier purely to call load_state() on it, since
        load_state() immediately overwrites whatever initialize()
        produced.
        """
        return jax.random.key(0)


    def __init__(self, *, seed: int):
        """Seeds a new RandomKeys sequence from an integer "seed", via
        jax.random.PRNGKey."""
        self._key = jax.random.PRNGKey(seed)


    def __iter__(self):
        """Returns self, so a RandomKeys instance can be used directly
        wherever an iterator is expected."""
        return self


    def __next__(self):
        """Splits the internal key into a new internal key and a
        subkey, advancing the sequence, and returns the subkey."""
        self._key, subkey = jax.random.split(self._key)
        return subkey


    def __pre_dictify__(self):
        """Hook used by solax.save(): converts the current internal
        JAX key to a plain NumPy array (JAX keys aren't themselves
        NumPy/JSON serializable), wrapped in a fresh RandomKeys so what
        gets saved is the current iteration state."""
        return init_from_key(np.array(self._key))


    def __post_undictify__(self):
        """Inverse of __pre_dictify__, used by solax.load(): converts
        the restored NumPy key back to a JAX array, resuming the
        sequence from its saved iteration state."""
        return init_from_key(jnp.array(self._key))
    
    
    
save_load_registry.register("RandomKeys", RandomKeys, init_from_key)