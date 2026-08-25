"""
Early stopping guard: a Guard implementation that wraps Flax's own
"EarlyStopping" class (flax.training.early_stopping.EarlyStopping),
watching a single scalar metric across successive updates and
signalling when training should stop, while also remembering the
NeuralModel's state at the best-seen value of that metric.
"""
from copy import copy
from flax.training.early_stopping import EarlyStopping

from .guard_base import *
from ..models import *


def get_flax_guard(params):
    """Builds a fresh Flax EarlyStopping instance, forwarding "params"
    directly as its constructor keyword arguments."""
    return EarlyStopping(**params)


class EarlyStoppingGuard(Guard):
    """
    Guard implementation delegating its bookkeeping to Flax's own
    EarlyStopping class. On each inform() call it feeds one watched
    scalar metric value to the underlying Flax EarlyStopping, and
    whenever that value is an improvement, it snapshots the current
    Flax state of "model" as the best one seen so far (see
    best_model). bool(guard) reports whether the underlying
    EarlyStopping says training should stop; motivation and semantics
    (patience, min_delta, etc.) are exactly Flax's own -- see SciPost
    Phys. Codebases 51 for how the parameters are forwarded.
    """

    def __init__(self,
                 model: NeuralModel,
                 *,
                 smaller_better: bool,
                 **early_stopping_params
                ):
        """
        Input:
            - "model": the NeuralModel whose Flax state is watched and
                snapshotted at each improvement.
            - "smaller_better": if True, a smaller metric value counts
                as an improvement (e.g. a loss); if False, a larger
                value does (e.g. an accuracy). Internally this is
                implemented by negating the watched value when
                "smaller_better" is False, since Flax's EarlyStopping
                always treats smaller as better.
            - "\*\*early_stopping_params": forwarded directly as keyword
                arguments to the underlying Flax EarlyStopping class
                (e.g. "patience", "min_delta").

        Calls reset() to initialize the underlying Flax guard and
        record "model"'s current Flax state as the (initial) best one.
        """
        self._model = model
        self.to_watched = (lambda x: x) if smaller_better else (lambda x: -x)
        self.params = early_stopping_params
        self.reset()


    def reset(self):
        """Rebuilds the underlying Flax EarlyStopping from scratch
        (using the stored constructor "params") and resets the
        best-seen Flax state to "model"'s current Flax state."""
        self._flax_guard = get_flax_guard(self.params)
        self._best_flax_state = self._model._flax_state


    def inform(self, data_entry):
        """
        Feeds one new metrics entry to the guard. "data_entry" must be
        a mapping with exactly one item (the single metric being
        watched); raises NotImplementedError otherwise, since
        multi-metric early stopping is not implemented. The entry's
        value is transformed via "to_watched" (negated unless
        "smaller_better" was True) and passed to the underlying Flax
        EarlyStopping's update(). If that update reports an
        improvement, the watched model's current Flax state is saved
        as the new best-seen state.
        """
        if len(data_entry) != 1:
            raise NotImplementedError("Early stopping implemented only for monitoring exactly one metrics.")
        val_orig = next(iter(data_entry.values()))
        val_watched = self.to_watched(val_orig)
        self._flax_guard = self._flax_guard.update(val_watched)
        if self._flax_guard.has_improved:
            self._best_flax_state = self._model._flax_state


    def __bool__(self):
        """True if the underlying Flax EarlyStopping says training
        should stop now (patience exhausted without improvement)."""
        return self._flax_guard.should_stop


    @property
    def steps_from_best(self) -> int:
        """Number of consecutive inform() calls since the last
        improvement (Flax EarlyStopping's "patience_count")."""
        return self._flax_guard.patience_count


    @property
    def best_model(self) -> NeuralModel:
        """A shallow copy of the watched model with its Flax state
        reset to the best-seen one recorded so far. The original
        model (and its current, possibly-worse state) is left
        untouched."""
        model_cpy = copy(self._model)
        model_cpy._flax_state = self._best_flax_state
        return model_cpy