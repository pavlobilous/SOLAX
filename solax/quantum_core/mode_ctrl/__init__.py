"""
mode_ctrl: global switches and context managers controlling
cross-cutting behavior across quantum_core -- how many determinants
get printed (printing) and whether duplicate determinants/rows are
auto-deduplicated ("squeezed") (squeezing).
"""
from .printing import dets_printing_limit
from .squeezing import manual_squeezing