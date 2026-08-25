"""
SOLAX's save/load subsystem: registers SOLAX classes for
(de)serialization (registration.py), converts registered objects and
plain nested dicts to/from plain dicts (dictification.py), and reads/
writes those dicts (with any NumPy arrays inside) to disk as JSON plus
".npy" files (json_for_dicts/). See "save"/"load" (usr_save_load.py)
for the user-facing entry points, and SciPost Phys. Codebases 51 Sec. 4
of the SOLAX paper for the overall design.
"""
from .registration import save_load_registry
from .usr_save_load import save, load