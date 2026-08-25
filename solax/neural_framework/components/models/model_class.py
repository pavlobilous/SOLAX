"""
NeuralModel: the generic, task-agnostic wrapper around a Flax-based
neural network -- construction from plain per-entry functions,
JIT-compiled train/predict steps, and Orbax-based checkpointing. This
is the base class behind the ready-made LeastSqRegressor/
SoftmaxClassifier (see ready_classes.py) and, at the solax interface
level, behind BasisClassifier (SciPost Phys. Codebases 51).
"""
import os
from collections.abc import Callable

from .flax_fundament import *
from ..jitted_core import *
from .orbax_save_load import *


class NeuralModel:
    """
    Wraps a user-defined architecture function together with a loss
    function and an optional output post-transform into a trainable,
    predictable, and savable/loadable neural network, backed by Flax.
    Before use, initialize() must be called to build the underlying
    Flax TrainState from a jax key, a batch of dummy features (for
    shape inference), and an optax optimizer; until then, "_flax_state"
    is None and train()/print_summary()/save_state()/load_state() are
    not usable (save_state()/load_state() raise IOError; the others
    would fail when the JIT-compiled functions try to use a None
    state).

    The functions needed for instantiating a Model are:

        call_on_entry(nn_inp) -> nn_out
        loss_fn(nn_out, label) -> loss
        post_transform(nn_out) -> nn_out_transformed

    Note:

        1. All functions deal with single (non-vectorized) entries.
        2. Outputs from loss_fn will be averaged over vectorized data.
    """

    def __init__(self, call_on_entry: Callable,
                       loss_fn: Callable,
                       post_transform: Callable = lambda x: x
                ):
        """
        Input:
            - "call_on_entry": per-entry architecture function
                (nn_inp) -> nn_out, e.g. a small Flax-friendly
                function or module call; for a classifier this should
                output raw logits, since solax applies softmax
                internally rather than requiring it in the
                architecture.
            - "loss_fn": per-entry loss function (nn_out, label) ->
                loss, batch-averaged internally during training (see
                get_trainer()).
            - "post_transform" (default=identity): per-entry function
                nn_out -> transformed output, applied after the
                forward pass at prediction time (see get_predictor()),
                e.g. argmax for a classifier.

        Builds the JIT-compiled trainer/predictor functions (see
        jitted_core.get_trainer/get_predictor) right away, but does
        not yet build the Flax state -- call initialize() for that.
        """
        self.call_on_entry = call_on_entry
        self.loss_fn = loss_fn
        self.post_transform = post_transform
        self._trainer = get_trainer(loss_fn)
        self._predictor = get_predictor(post_transform)
        self._flax_state = None


    def initialize(self, key, dummy_features, optimizer):
        """
        Builds the underlying Flax TrainState (and its tabulated
        architecture summary) from "key" (for parameter
        initialization), "dummy_features" (a batch used only to trace
        parameter shapes; its values are irrelevant), and "optimizer"
        (an optax optimizer). Sets "self._flax_state" and
        "self._summary". Must be called once before train()/
        __call__()/save_state()/load_state() are used.
        """
        self._flax_state, self._summary = create_state_and_summary(
            key, self.call_on_entry, dummy_features, optimizer
        )


    def __call__(self, features):
        """Runs the forward pass on a batch of "features" using the
        current Flax state, applying "post_transform" to each entry of
        the output. Returns the batch of (transformed) predictions."""
        return self._predictor(self._flax_state, features)


    def train(self, features, labels):
        """Performs one gradient-descent optimizer step on a batch of
        (features, labels), updating "self._flax_state" in place (by
        reassignment) to the state after the step."""
        self._flax_state = self._trainer(self._flax_state, features, labels)


    def print_summary(self):
        """Prints the tabulated Flax architecture summary produced at
        initialize() time."""
        print(self._summary)


    def save_state(self, fld: str):
        """Saves the current Flax state (parameters and optimizer
        state) to directory "fld" via Orbax (see
        orbax_save_load.save_flax_state), overwriting any existing
        contents there. Raises IOError if this model has not been
        initialize()d yet."""
        if self._flax_state is not None:
            save_flax_state(fld, self._flax_state)
        else:
            raise IOError('Cannot save state of an unitialized NeuralModel.')


    def load_state(self, fld: str):
        """
        Loads a previously saved Flax state from directory "fld" into
        this model's current Flax state (see
        orbax_save_load.load_flax_state), replacing
        "self._flax_state". The model must already be initialize()d
        (its current state is used as the pytree structure/dtype
        template Orbax restores into) -- typically initialize() is
        called with an arbitrary/throwaway key beforehand, since the
        loaded parameters will overwrite whatever it produced. Raises
        FileNotFoundError if "fld" does not exist, or IOError if this
        model has not been initialize()d yet.
        """
        if not os.path.exists(fld):
            raise FileNotFoundError("Cannot load from here. Path does not exist.")
        if self._flax_state is not None:
            self._flax_state = load_flax_state(fld, self._flax_state)
        else:
            raise IOError('Cannot load state for an unitialized NeuralModel. Call "initialize" first and try again.')