"""
MetricsMonitor: keeps a rolling history of one or more named metrics
values over the course of training/validation, optionally evaluating
them itself (via a JIT-compiled validator built from per-entry metrics
functions), optionally printing/reporting each update, and optionally
driving an early stopping guard.
"""
from collections import deque
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
    Tracks the history of one or more named metrics as a dict of
    fixed-maxlen deques (most recent entry first), one deque per
    metric name. Optionally owns the means to evaluate those metrics
    itself from raw (features, labels) batches, to report each update
    (see the "reporting"/"reporting_updates" context managers, which
    register/unregister a Reporter under UPDATE_REPORTER_KEY), and to
    inform an early stopping Guard of each update.

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
        """
        Input:
            - "metrics_fns" (default=None): optional dict mapping each
                metric name to a per-entry "metrics_fn(nn_out, label)
                -> value" function. If given, this monitor becomes
                "self-evaluating": calling it (see __call__) forwards
                to a JIT-compiled validator built from these functions
                and "model"'s current Flax state; also determines the
                tracked metric names (overriding "names").
            - "model" (default=None): the NeuralModel whose Flax state
                the validator reads from; required (implicitly, via
                "metrics_fns" being usable) whenever "metrics_fns" is
                given.
            - "names" (default=None): names of the metrics to track,
                used only when "metrics_fns" is not given (e.g. for a
                monitor that is only ever fed externally-computed
                entries via update()).
            - "maxlen" (default=None): maximum number of past entries
                kept per metric (oldest entries are dropped once
                exceeded); None means unbounded.
            - "early_stopping" (default=None): a Guard informed of
                every update() call. If omitted, self.early_stopping
                returns an always-False DummyGuard instead.
        """
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
        """The configured early-stopping Guard, or a fresh DummyGuard
        (always bool()==False) if none was configured."""
        guard = self._guards.get(EARLY_STOPPING_GUARD_KEY, DummyGuard())
        return guard


    def get_data_prototype(self,
                           *,
                           names: Iterable[str] = None,
                           maxlen: int | None = MAXLEN_ABSENT
                          ):
        """
        Builds a fresh dict of empty deques, one per name in "names"
        (or, if "names" is None, one per metric name currently
        tracked by "self"), each with the given "maxlen" (or, if
        omitted, "self"'s own "maxlen"). Used both to initialize
        "self._data" and, by the "aggregating" context manager, to
        build an isolated data store with the same metric names.
        """
        if names is None:
            names = self._data
        if maxlen is MAXLEN_ABSENT:
            maxlen = self.maxlen
        return {name : deque(maxlen=maxlen) for name in names}


    def __call__(self, features, labels):
        """
        Evaluates the tracked metrics on a raw (features, labels)
        batch, using the JIT-compiled validator built from
        "metrics_fns" at construction time. Returns the dict of metric
        name -> batch-averaged value, without recording it (see
        update()/eval_and_update() to also record it). Raises
        AttributeError if this monitor was constructed without
        "metrics_fns" (no self-evaluation means).
        """
        if self._validator is None:
            raise AttributeError("This MetricsMonitor does not have its own "\
                                 "metrics evaluation means.")
        new_entry = self._validator(features, labels)
        return new_entry


    def update(self, new_entry: dict[str, float],
               *, report_label=""):
        """
        Records one already-computed metrics entry: appends each
        "new_entry" value to the front of its metric's deque (so the
        most recent entry is always first), forwards the entry to the
        active update reporter (if any; see the "reporting_updates"
        context manager) tagged with "report_label", and informs the
        early stopping guard (self.early_stopping) of the entry.
        """
        for name in self._data:
            self._data[name].appendleft(new_entry[name])
        self.submit_report(UPDATE_REPORTER_KEY, new_entry, report_label=report_label)
        self.early_stopping.inform(new_entry)


    def eval_and_update(self, features, labels,
                        *, report_label=""):
        """Evaluates the tracked metrics on (features, labels) (see
        __call__) and records the result via update()."""
        new_entry = self(features, labels)
        self.update(new_entry, report_label=report_label)


    def clear(self):
        """Empties the recorded history of every tracked metric and
        resets the early stopping guard (self.early_stopping.reset()),
        as if this monitor were freshly constructed."""
        for vals in self._data.values():
            vals.clear()
        self.early_stopping.reset()


    @property
    def maxlen(self):
        """The maxlen shared by all metric deques (as configured at
        construction)."""
        return next(iter(self._data.values())).maxlen


    def __len__(self):
        """Number of tracked metrics (not the length of any one
        metric's recorded history -- see __getitem__ for that)."""
        return len(self._data)


    def __getitem__(self, s):
        """
        Indexes/slices into each metric's recorded history, returning
        a dict of metric name -> jax array of the selected entries.
        "s" is an int or a slice, applied identically to every
        metric's deque.
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
        Reduces the full recorded history of every tracked metric to a
        single value via "reducing_func" (e.g. jnp.mean). Returns a
        dict of metric name -> reduced value.
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
        """The mean of each tracked metric's full recorded history, as
        a dict of metric name -> mean value. Shorthand for
        reduced(jnp.mean)."""
        return self.reduced(jnp.mean)



    def submit_report(self,
                      reporter_key: Hashable,
                      data_entry: dict[str, float],
                      *, report_label=""):
        """
        Forwards "data_entry" to the reporter registered under
        "reporter_key" (if any is currently active, e.g. via the
        "reporting"/"reporting_updates" context managers), tagged with
        "report_label". Does nothing if no reporter is registered
        under that key.
        """
        reporter = self._reporters.get(reporter_key)
        if reporter:
            reporter.make_report(data_entry, report_label)