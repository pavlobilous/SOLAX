from collections import deque
from itertools import chain
import jax.numpy as jnp
from collections.abc import Iterable, Callable
from typing import Hashable
from numbers import Integral

from ..jitted_core import *
from ..models import *
from ..guards import *


MAXLEN_ABSENT = object()
UPDATE_REPORTER_KEY = object()
EARLY_STOPPING_GUARD_KEY = object()


class MetricsMonitor:
    """
    The (optional) functions for Metrics instantiation are:
        metrics_fn(nn_out, label) -> metrics
    Note:
        1. They deal with single (non-vectorized) entries.
        2. Their outputs will be averaged over vectorized data.
    """

    def __init__(self,
                 metrics_fns: dict[str, Callable] = None,
                 model: NeuralModel = None,
                 *,
                 names: Iterable[str] = None,
                 maxlen: int | None = None,
                 early_stopping: Guard = None
                ):
        self.metrics_fns = metrics_fns
        if metrics_fns:
            names = metrics_fns.keys()
        self._data = self.get_data_prototype(names=names, maxlen=maxlen)
        if metrics_fns is not None:
            validator = get_validator(metrics_fns)
            self._validator = lambda features, labels: validator(
                                model._flax_state, features, labels)
        else:
            self._validator = None
        self._reporters = {}
        self._guards = {}
        if early_stopping is not None:
            self._guards[EARLY_STOPPING_GUARD_KEY] = early_stopping
            
            
    @property
    def early_stopping(self):
        guard = self._guards.get(EARLY_STOPPING_GUARD_KEY, DummyGuard())
        return guard
        
        
    def get_data_prototype(self,
                           *,
                           names: Iterable[str] = None,
                           maxlen: int | None = MAXLEN_ABSENT
                          ):
        if names is None:
            names = self._data
        if maxlen is MAXLEN_ABSENT:
            maxlen = self.maxlen
        return {name : deque(maxlen=maxlen) for name in names}
    
    
    def __call__(self, features, labels):
        if self._validator is None:
            raise AttributeError("This MetricsMonitor does not have its own "\
                                 "metrics evaluation means.")
        new_entry = self._validator(features, labels)
        return new_entry

      
    def update(self, new_entry: dict[str, float],
               *, report_label=""):
        for name in self._data:
            self._data[name].appendleft(new_entry[name]) 
        self.submit_report(UPDATE_REPORTER_KEY, new_entry, report_label=report_label)
        self.early_stopping.inform(new_entry)
            
            
    def eval_and_update(self, features, labels,
                        *, report_label=""):
        new_entry = self(features, labels)
        self.update(new_entry, report_label=report_label)
        
        
    def clear(self):
        for vals in self._data.values():
            vals.clear()
        self.early_stopping.reset()
        
        
    @property
    def maxlen(self):
        return next(iter(self._data.values())).maxlen
        
        
    def __len__(self):
        return len( next(iter(self._data)) )
    
    
    def __getitem__(self, s):
        """
        Note: The last updates come first.
        """
        if isinstance(s, (Integral, slice)):
            entries = {
                name: jnp.array(vals)[s]
                for name, vals in self._data.items()
            }
            return entries
        else:
            return NotImplemented
    
        
    def reduced(self, reducing_func: Callable[[jnp.ndarray], float]):
        """
        Note: The last updates come first.
        """
        entries = self[:]
        reduced = {
            name: reducing_func(arr)
            for name, arr in entries.items()
        }
        return reduced
    
    
    @property
    def average(self):
        return self.reduced(jnp.mean)
        
    

    def submit_report(self,
                      reporter_key: Hashable,
                      data_entry: dict[str, float],
                      *, report_label=""): 
        reporter = self._reporters.get(reporter_key)
        if reporter:
            reporter.make_report(data_entry, report_label)