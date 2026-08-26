json_for_dicts
==============

Low-level (de)serialization of a nested dict (with NumPy arrays inside)
to/from disk -- the on-disk half of :func:`solax.save`/:func:`solax.load`,
combined with :doc:`dictification` (which turns registered solax
objects into plain nested dicts first).

dumper
------

.. automodule:: solax.save_load.json_for_dicts.dumper
   :members:
   :undoc-members:
   :show-inheritance:

loader
------

.. automodule:: solax.save_load.json_for_dicts.loader
   :members:
   :undoc-members:
   :show-inheritance:
