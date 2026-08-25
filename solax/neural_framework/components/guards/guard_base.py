"""
Abstract interface for training-loop stopping guards, plus a no-op
fallback implementation.
"""
from abc import ABC, abstractmethod


class Guard(ABC):
    """
    Abstract base class for a stopping condition a MetricsMonitor can
    consult during training. A Guard is fed metrics data via inform()
    and reports whether training should stop via bool(guard); reset()
    clears any accumulated state (e.g. at the start of a new training
    run). See EarlyStoppingGuard for the concrete early-stopping
    implementation, and DummyGuard for the always-False fallback used
    when no guard is configured.
    """

    @abstractmethod
    def reset(self) -> None:
        """Clears any accumulated state, as if freshly constructed."""
        pass

    @abstractmethod
    def inform(self, data) -> None:
        """Feeds one new data entry (as produced by MetricsMonitor
        updates) to the guard, updating its internal state."""
        pass

    @abstractmethod
    def __bool__(self):
        """True if, based on the data seen so far, training should
        stop now."""
        pass



class DummyGuard(Guard):
    """
    No-op Guard: reset()/inform() do nothing and bool() is always
    False, so training never stops early. Used as the default guard
    (e.g. via MetricsMonitor.early_stopping) when no real guard was
    configured, so callers can use a guard unconditionally without
    checking for None.
    """

    def reset(self) -> None:
        """Does nothing."""
        pass

    def inform(self, data):
        """Does nothing."""
        pass

    def __bool__(self):
        """Always False."""
        return False