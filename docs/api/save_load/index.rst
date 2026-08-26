save_load
=========

Internals of solax's save/load subsystem, beyond the user-facing
:func:`solax.save`/:func:`solax.load` already documented on the
:doc:`../solax` page: the class registry, the object-to-dict translation
layer, and the on-disk (de)serialization. See also the "For developers"
section on the front page for a guide to making a custom class
savable/loadable.

.. toctree::
   :maxdepth: 1

   registration
   dictification
   json_for_dicts
