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
    
    with val_rep(f"Epoch"), train_rep():
        for ep in range(epochs):
            key, subkey = jax.random.split(key)
            train(subkey, *train_data)
            if val_metrics:
                with averaging(val_metrics, report_label=ep):
                    key, subkey = jax.random.split(key)
                    validate(subkey, *val_data)
            if val_metrics.early_stopping:
                early_stopped = True
                break
        else:
            early_stopped = False
            
    return early_stopped