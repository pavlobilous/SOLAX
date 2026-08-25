"""
Helpers for splitting data into batches, shuffling indices, and
aggregating per-batch metrics.
"""
from .aggregating import aggregating, averaging
from .batchification import batchify, exhaust_batches
from .index_shuffle import shuffled_inds