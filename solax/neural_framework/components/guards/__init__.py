"""
Guards: pluggable stopping conditions that a MetricsMonitor can
consult during training (see guard_base.Guard and its concrete
implementations DummyGuard and EarlyStoppingGuard).
"""
from .guard_base import Guard, DummyGuard
from .early_stopping import EarlyStoppingGuard