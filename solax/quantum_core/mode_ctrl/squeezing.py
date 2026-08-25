"""
Global switches controlling when Basis/State/OperatorTerm automatically
"squeeze" (deduplicate/merge) their rows, and the manual_squeezing()
context manager for suspending that behavior.
"""
from contextlib import contextmanager
from copy import deepcopy


squeeze_params = {
    "SQUEEZE_BASIS_AFTER_INIT": True,
    "SQUEEZE_OPTERM_AFTER_INIT": True,
    "SQUEEZE_DETS_AFTER_ADD": True,
    "SQUEEZE_OPTERM_AFTER_ADD": True,
    "SQUEEZE_DETS_AFTER_OPTERM": True
}


@contextmanager
def manual_squeezing():
    """
    Context manager that temporarily disables all automatic
    "squeezing" (duplicate-determinant/duplicate-row deduplication) on
    Basis/State/OperatorTerm construction, addition, and OperatorTerm
    application, restoring the previous squeeze_params on exit. Used
    internally wherever an intermediate result must be built without
    eagerly deduplicating it (e.g. while assembling a larger result in
    stages); "squeeze"/deduplication is solax's own internal naming,
    not paper terminology. Use squeeze()/is_squeezed on the individual
    classes to deduplicate/check explicitly regardless of these
    settings.
    """
    try:
        squeeze_params_old = deepcopy(squeeze_params)
        squeeze_params.update(dict.fromkeys(squeeze_params, False))
        yield
    finally:
        squeeze_params.update(squeeze_params_old)