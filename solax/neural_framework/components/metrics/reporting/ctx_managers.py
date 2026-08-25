"""
Context managers that temporarily attach a Reporter to a
MetricsMonitor, so that calls made while inside the "with" block (e.g.
update()) print (or capture) a line per call, tagged with a prefix and
per-call label.
"""
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
    Input:
        - "metrics_monitor": the MetricsMonitor to attach a reporter
            to. If None, this is a no-op (nullcontext): nothing is
            reported.
        - "reporter_key": key identifying which reporter slot to
            occupy on "metrics_monitor" (e.g. UPDATE_REPORTER_KEY, so
            that submit_report() calls tagged with that key are
            picked up). Raises RuntimeError if a reporter is already
            active under this key.
        - "prefix" (default=""): text prepended (with a space) to
            each reported line's label.
        - "stdout" (default=True): if True, reports are printed to
            sys.stdout as they happen and the context yields None; if
            False, reports are written to an in-memory Report buffer
            instead (nothing is printed) and the context yields that
            buffer, so the caller can inspect its contents (e.g. via
            repr()) after (or during) the "with" block.
    On exit (even if the block raised), the reporter is unregistered
    from "metrics_monitor", regardless of "stdout".
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
    Context manager for reporting updates to the "metrics_monitor"
    data. Shorthand for reporting() with "reporter_key" set to
    UPDATE_REPORTER_KEY, so that every update() call on
    "metrics_monitor" made inside the "with" block is reported; see
    reporting() for the full behavior (including the
    "metrics_monitor=None" no-op case and the "stdout" buffer option).
    """
    return reporting(metrics_monitor, UPDATE_REPORTER_KEY, prefix=prefix, stdout=stdout)