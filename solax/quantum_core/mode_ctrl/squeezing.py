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
    try:
        squeeze_params_old = deepcopy(squeeze_params)
        squeeze_params.update(dict.fromkeys(squeeze_params, False))
        yield
    finally:
        squeeze_params.update(squeeze_params_old)