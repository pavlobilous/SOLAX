from contextlib import contextmanager

print_params = {
    "DETS_PRINTING_LIMIT": 5
}


@contextmanager
def dets_printing_limit(limit: int | None):
    try:
        old_limit = print_params["DETS_PRINTING_LIMIT"]
        print_params["DETS_PRINTING_LIMIT"] = limit
        yield
    finally:
        print_params["DETS_PRINTING_LIMIT"] = old_limit