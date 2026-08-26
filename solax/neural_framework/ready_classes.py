"""
Ready-made NeuralModel/MetricsMonitor subclasses exposed as the two
built-in loss/metric configurations described in SciPost Phys.
Codebases 51: a least-squares regressor and a softmax classifier, each
paired with a matching metrics monitor (loss and accuracy,
respectively).
"""
import jax.numpy as jnp
import optax

from .components import *


class LeastSqRegressor(NeuralModel):
    """
    NeuralModel preconfigured for least-squares regression: per-entry
    loss is the raw squared error (x - y)**2 (batch-averaged, like any
    NeuralModel loss). "call_on_entry" should output a raw prediction
    matching the shape/scale of the labels; there is no output
    post-transform (identity). Matches the paper's "square loss"
    l2(y, y_pred) = 0.5*(y - y_pred)^2 up to the constant factor 0.5,
    which does not affect the location of the minimum reached by
    gradient descent.
    """

    def __init__(self, call_on_entry):
        """Wraps "call_on_entry" (the per-entry architecture function)
        into a NeuralModel with squared-error loss."""
        super().__init__(call_on_entry,
                         lambda x, y: (x - y)**2 )


class SoftmaxClassifier(NeuralModel):
    """
    NeuralModel preconfigured for classification: per-entry loss is
    softmax cross-entropy against integer class labels
    (optax.softmax_cross_entropy_with_integer_labels), and the output
    post-transform is argmax (so calling the model returns predicted
    class indices, not raw logits/probabilities). "call_on_entry"
    should output raw logits -- solax applies softmax internally, so
    there is no need to apply it within the architecture function
    itself (SciPost Phys. Codebases 51).
    """

    def __init__(self, call_on_entry):
        """Wraps "call_on_entry" (the per-entry architecture function,
        expected to output raw logits) into a NeuralModel with softmax
        cross-entropy loss and argmax post-transform."""
        super().__init__(call_on_entry,
                         optax.softmax_cross_entropy_with_integer_labels,
                         jnp.argmax)


class LossMonitor(MetricsMonitor):
    """
    MetricsMonitor preconfigured to track a single metric, "loss",
    evaluated via "model.loss_fn" (so this only makes sense paired
    with the same model whose loss it tracks).
    """
    def __init__(self, model: NeuralModel,
                 *, early_stopping: Guard = None):
        """
        Input:
            - "model": the NeuralModel to evaluate/track; its
                "loss_fn" is used as the tracked metric.
            - "early_stopping" (default=None): optional Guard informed
                of every update (see MetricsMonitor).
        """
        metrics_fns = {"loss": model.loss_fn}
        super().__init__(metrics_fns, model, early_stopping=early_stopping)


class AccuracyMonitor(MetricsMonitor):
    """
    MetricsMonitor preconfigured to track a single metric, "accuracy":
    the fraction of entries where "model"'s post-transformed output
    exactly matches the label (as 0/1 per entry, batch-averaged into a
    fraction by the underlying validator). Intended for classifiers
    (e.g. SoftmaxClassifier), where post_transform is argmax and
    labels are integer class indices.
    """

    def __init__(self, model: NeuralModel,
                 *, early_stopping: Guard = None):
        """
        Input:
            - "model": the NeuralModel to evaluate/track; its
                "post_transform" is applied to raw network outputs
                before comparing them to labels.
            - "early_stopping" (default=None): optional Guard informed
                of every update (see MetricsMonitor).
        """
        def accuracy(x, y):
            x = model.post_transform(x)
            return (x == y).astype(int)

        metrics_fns = {"accuracy": accuracy}
        super().__init__(metrics_fns, model, early_stopping=early_stopping)