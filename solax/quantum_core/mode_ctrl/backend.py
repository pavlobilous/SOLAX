from contextlib import contextmanager

backend_params = {
    "ENGINE": "jax"
}


@contextmanager
def op_action_via_numpy():
    try:
        old_engine = backend_params["ENGINE"]
        backend_params["ENGINE"] = "numpy"
        yield
    finally:
        backend_params["ENGINE"] = old_engine