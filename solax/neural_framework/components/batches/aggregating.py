"""
Context managers that temporarily redirect a MetricsMonitor's updates
into a fresh, isolated history, then collapse ("reduce") that history
into a single entry and feed it back as one update to the original
monitor. Used to turn many per-batch metrics updates (e.g. one per
validation batch) into a single per-epoch summary (e.g. the epoch's
average).
"""
import jax.numpy as jnp
from collections.abc import Callable
from contextlib import contextmanager

from ..metrics import *
from ..ctx_helpers import *


@null_if_arg0_none
@contextmanager
def aggregating(metrics_monitor: MetricsMonitor,
                reducing_func: Callable[[jnp.ndarray], float],
                *,
                maxlen: int | None = None,
                report_label=""
            ):
    """
    Context manager for aggregating data before updating the MetricsMonitor.
    Input:
        - "metrics_monitor": the MetricsMonitor to aggregate updates for.
            If None, this is a no-op (nullcontext): "metrics_monitor" is
            used and updated as usual, without aggregation.
        - "reducing_func": function reducing a 1D jax array of values
            (across all entries recorded inside the "with" block) to a
            single float, e.g. jnp.mean.
        - "maxlen" (default=None): maxlen passed to the temporary,
            isolated data store used while inside the "with" block.
        - "report_label" (default=""): label passed to the single,
            reduced update that is submitted to "metrics_monitor" on
            exit.
    While inside the "with" block, "metrics_monitor"'s data, reporters,
    and guards (including early stopping) are swapped out for empty
    ones, so any calls to update()/eval_and_update() during the block
    accumulate in an isolated history and do not reach reporters or
    guards. On exit (even if the block raised), the original data/
    reporters/guards are restored and a single reduced entry -- each
    metric's isolated history passed through "reducing_func" -- is
    submitted via update(), so it does reach the real reporters and
    guards exactly once.
    """
    try:
        data_old = metrics_monitor._data
        metrics_monitor._data = metrics_monitor.get_data_prototype(maxlen=maxlen)
        reporters_old = metrics_monitor._reporters
        metrics_monitor._reporters = {}
        guards_old = metrics_monitor._guards
        metrics_monitor._guards = {}
        yield
    finally:
        aggregated_entry = metrics_monitor.reduced(reducing_func)
        metrics_monitor._data = data_old
        metrics_monitor._reporters = reporters_old
        metrics_monitor._guards = guards_old
        metrics_monitor.update(aggregated_entry, report_label=report_label)
    
    
def averaging(metrics_monitor: MetricsMonitor,
              *,
              maxlen: int | None = None,
              report_label=""):
    """
    Context manager for averaged aggregating data before updating the
    MetricsMonitor. Shorthand for aggregating() with "reducing_func"
    set to jnp.mean; see aggregating() for the full behavior (including
    the "metrics_monitor=None" no-op case).
    """
    return aggregating(metrics_monitor, jnp.mean, maxlen=maxlen, report_label=report_label)