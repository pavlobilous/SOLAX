"""
Metrics tracking during training/validation: MetricsMonitor stores a
rolling history of metrics values and optionally drives an early
stopping guard; reporting provides context managers for printing/
capturing metrics updates as they happen.
"""
from .monitor_class import MetricsMonitor
from .reporting import *