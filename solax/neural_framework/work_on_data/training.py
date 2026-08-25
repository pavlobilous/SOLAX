"""
The generic, task-agnostic training loop driving a NeuralModel over a
full (already train/validation-split) dataset: batched gradient
updates per epoch, optional periodic training-metrics reporting,
optional per-epoch validation with early stopping, and printed
progress (per SciPost Phys. Codebases 51's description of the
training procedure: batches, epochs, a held-out validation split,
accuracy/loss printed per epoch).
"""
import jax
from collections.abc import Sequence

from ..components import *


Data = tuple[Sequence, Sequence]

def train_on_data(key,
                  model: NeuralModel,
                  train_data: Data,
                  *,
                  val_data: Data = None,
                  batch_size: int | None = None,
                  epochs: int = 1,
                  train_metrics: MetricsMonitor = None,
                  val_metrics: MetricsMonitor = None,
                  train_metr_freq: int = 10,
                  val_at_start: bool = False,
                  printout_vals: bool = True
        ):
    """
    Trains "model" on "train_data" for up to "epochs" epochs, with
    optional per-batch/per-epoch metrics tracking and early stopping.

    Input:

        - "key": jax.random key; split as needed to shuffle each
            epoch's training batches and each validation pass.
        - "model": the NeuralModel to train (mutated in place via its
            own train() calls).
        - "train_data": (features, labels) pair for training, batched
            per "batch_size" and reshuffled every epoch.
        - "val_data" (default=None): (features, labels) pair for
            validation; required whenever "val_metrics" is given (see
            Note below).
        - "batch_size" (default=None): batch size for both training
            and validation batching; None means a single batch.
        - "epochs" (default=1): number of passes over "train_data".
        - "train_metrics" (default=None): optional MetricsMonitor
            evaluated (and updated, and reported) on training batches
            every "train_metr_freq" batches; if None, no training
            metrics are evaluated during the loop.
        - "val_metrics" (default=None): optional MetricsMonitor
            evaluated over all of "val_data" (averaged into one
            per-epoch entry, see the "averaging" context manager)
            after every epoch, and once more before the first epoch if
            "val_at_start" is True. Also consulted for early stopping
            at the end of every epoch (see Note below).
        - "train_metr_freq" (default=10): evaluate/report
            "train_metrics" every this many training batches (batch 0
            included).
        - "val_at_start" (default=False): if True (and "val_metrics"
            is given), run one extra validation pass before the first
            training epoch, to record/report the untrained model's
            baseline metrics.
        - "printout_vals" (default=True): if True, metrics updates
            (training and validation) are printed to stdout as they
            happen; if False, they are computed/tracked silently.

    Output:
        A bool: True if training stopped early because
        "val_metrics.early_stopping" signalled to stop (see
        EarlyStoppingGuard) before all "epochs" completed; False if
        all "epochs" ran to completion.
    """
    @exhaust_batches
    @batchify(batch_sz=batch_size, shuffle=True)
    def train(i, features, labels):
        model.train(features, labels)
        if train_metrics and (i % train_metr_freq == 0):
            rl = f"  Batch {i}"
            if i == 0:
                rl = "\n" + rl
            train_metrics.eval_and_update(features, labels, report_label=rl)
    
    @exhaust_batches
    @batchify(batch_sz=batch_size, shuffle=True)
    def validate(i, features, labels):
        val_metrics.eval_and_update(features, labels)

    val_rep = lambda p: reporting_updates(val_metrics, prefix=p, stdout=printout_vals)
    train_rep = lambda: reporting_updates(train_metrics, stdout=printout_vals)
    
    with val_rep("Started"):
        if val_metrics and val_at_start:
            with averaging(val_metrics):
                key, subkey = jax.random.split(key)
                validate(subkey, *val_data)
    
    with val_rep("Epoch"), train_rep():
        for ep in range(epochs):
            key, subkey = jax.random.split(key)
            train(subkey, *train_data)
            if val_metrics:
                with averaging(val_metrics, report_label=ep):
                    key, subkey = jax.random.split(key)
                    validate(subkey, *val_data)
            if val_metrics and val_metrics.early_stopping:
                early_stopped = True
                break
        else:
            early_stopped = False
            
    return early_stopped