save_load
=========

solax's save/load subsystem. ``save``/``load`` themselves are shown on
the :doc:`Public API <../../../solax>` page (re-exported as
``sx.save``/``sx.load``); everything below is the machinery behind
them -- the class registry, the object-to-dict translation layer, and
the on-disk (de)serialization. See also the "For developers" section
on the front page for a guide to making a custom class savable/loadable.

.. toctree::
   :maxdepth: 1

   registration
   dictification
   json_for_dicts
   usr_save_load
