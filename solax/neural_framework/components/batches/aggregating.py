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
    Context manager for averaged aggregating data before updating the MetricsMonitor.
    """
    return aggregating(metrics_monitor, jnp.mean, maxlen=maxlen, report_label=report_label)