import sys
from typing import Hashable
from contextlib import contextmanager

from .report_classes import *
from ..monitor_class import *
from ...ctx_helpers import *


@null_if_arg0_none
@contextmanager
def reporting(metrics_monitor: MetricsMonitor,
              reporter_key: Hashable,
              *,
              prefix: str = "",
              stdout: bool = True
             ):
    """
    Context manager for reporting from methods of "metrics_monitor".
    """
    if reporter_key in metrics_monitor._reporters:
        raise RuntimeError("This reporter is already active.")
    try:
        buffer = Report() if not stdout else sys.stdout
        metrics_monitor._reporters[reporter_key] = Reporter(buffer, prefix)
        yield buffer if not stdout else None
    finally:
        del metrics_monitor._reporters[reporter_key]
        
                  
def reporting_updates(metrics_monitor: MetricsMonitor,
                      *,
                      prefix: str = "",
                      stdout: bool = True
            ):
    """
    Context manager for reporting updates to the "metrics_monitor" data.
    """
    return reporting(metrics_monitor, UPDATE_REPORTER_KEY, prefix=prefix, stdout=stdout)