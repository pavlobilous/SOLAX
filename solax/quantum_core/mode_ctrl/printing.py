"""
Global switch controlling how many determinants Basis.__str__/
State.__str__ render before truncating with "...".
"""
from contextlib import contextmanager

print_params = {
    "DETS_PRINTING_LIMIT": 5
}


@contextmanager
def dets_printing_limit(limit: int | None):
    """
    Context manager that temporarily sets the maximum number of
    determinants shown by Basis.__str__/State.__str__ (and the
    print()/str() output built on them) to "limit", restoring the
    previous limit on exit. Pass None for no limit (render every
    determinant); the default limit is 5.
    """
    try:
        old_limit = print_params["DETS_PRINTING_LIMIT"]
        print_params["DETS_PRINTING_LIMIT"] = limit
        yield
    finally:
        print_params["DETS_PRINTING_LIMIT"] = old_limit