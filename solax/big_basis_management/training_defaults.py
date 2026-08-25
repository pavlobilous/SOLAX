"""
Default keyword arguments for BigBasisManager.train_classifier (see
manager_class.py), forwarded to train_on_data. Callers of
train_classifier override individual entries via its "batch_size"/
"epochs"/**train_kwargs arguments; anything not overridden falls back
to the value here.
"""

DEFAULT_TRAIN_KWARGS = dict(
    val_frac=0.2,           # fraction of the training State held out for validation
    train_metrics=None,     # MetricsMonitor for the training split (None: not tracked)
    train_metr_freq=10,     # report training metrics every this many batches
    val_at_start=True,      # evaluate validation metrics once before the first epoch
    printout_vals=True      # print metrics updates to stdout as training proceeds
)