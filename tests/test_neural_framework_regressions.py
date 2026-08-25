"""Regression tests for three latent bugs found while documenting
solax/neural_framework/ (none were reachable through any existing
caller, which is exactly why they went unnoticed -- see the fixes in
index_shuffle.py, monitor_class.py, and training.py).
"""
import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
import pytest

from solax.neural_framework.components.batches.index_shuffle import shuffled_inds
from solax.neural_framework import LossMonitor, LeastSqRegressor, train_on_data


def test_shuffled_inds_multi_chunk_path():
    """shuffled_chunks() used to read an undefined module-level
    "max_ind", raising NameError as soon as chunking actually kicked in
    (chunks_num > 1). Force that path with a small max_ind and check
    the result is still a valid permutation."""
    key = jax.random.PRNGKey(0)
    length = 17
    result = shuffled_inds(key, length=length, max_ind=5)
    assert sorted(result.tolist()) == list(range(length))


def test_metrics_monitor_len_is_metric_count():
    """MetricsMonitor.__len__ used to return len() of the first
    metric's *name string* rather than the number of tracked metrics."""
    monitor = LossMonitor.__new__(LossMonitor)
    monitor._data = {"loss": [], "accuracy": []}
    assert len(monitor) == 2


@pytest.mark.slow
def test_train_on_data_without_val_metrics_does_not_crash():
    """train_on_data used to unconditionally check
    val_metrics.early_stopping even when val_metrics was None, raising
    AttributeError instead of simply skipping early stopping."""

    class TinyModel(nn.Module):
        @nn.compact
        def __call__(self, x):
            return nn.Dense(1)(x)

    model = LeastSqRegressor(TinyModel())
    key = jax.random.PRNGKey(0)
    key, init_key = jax.random.split(key)
    dummy_features = jnp.zeros((1, 3))
    model.initialize(init_key, dummy_features, optax.adam(1e-2))

    features = jnp.array(np.random.default_rng(0).normal(size=(8, 3)))
    labels = features.sum(axis=1, keepdims=True)

    early_stopped = train_on_data(
        key, model, (features, labels),
        batch_size=4, epochs=2,
        train_metrics=LossMonitor(model),
        val_metrics=None,
        printout_vals=False,
    )
    assert early_stopped is False
