"""
The thin Flax linen.Module wrapper turning a user-supplied, per-entry
architecture function into a batched Flax module, plus the helper that
initializes it into a Flax TrainState.
"""
from collections.abc import Callable
import jax
from flax import linen as nn
from flax.training.train_state import TrainState


class Module(nn.Module):
    """
    Flax linen.Module wrapping a user-supplied, per-entry architecture
    function "call_on_entry(nn_inp) -> nn_out" (raw logits for a
    classifier, per SciPost Phys. Codebases 51 -- solax applies
    softmax itself, so "call_on_entry" should not). Calling the module
    on a batch vmaps "call_on_entry" over the batch dimension, so
    "call_on_entry" only ever needs to handle a single, non-vectorized
    entry.
    """
    call_on_entry: Callable

    @nn.compact
    def __call__(self, features):
        """Applies "call_on_entry" to every entry of "features" (its
        leading axis), via jax.vmap."""
        call_on_data = jax.vmap(self.call_on_entry)
        return call_on_data(features)


def create_state_and_summary(key, call_on_entry, dummy_features, optimizer):
    """
    Builds and initializes a Flax TrainState for a NeuralModel.
    Input:
        - "key": jax.random key used for parameter initialization.
        - "call_on_entry": the user's per-entry architecture function,
            wrapped in a Module (see Module).
        - "dummy_features": a batch of features with the expected
            shape/dtype, used only to trace the module's parameter
            shapes (its values do not matter) and to produce the
            tabulated summary.
        - "optimizer": an optax optimizer (e.g. optax.adam(...)) used
            as the TrainState's gradient transformation.
    Output:
        Tuple (state, summary): "state" is the initialized Flax
        TrainState (bundling the module's apply function, its
        initialized parameters, and the optimizer); "summary" is the
        module's tabulated architecture summary (a string produced by
        Module.tabulate), for NeuralModel.print_summary().
    """
    module = Module(call_on_entry)
    params = module.init(key, dummy_features)['params']
    state = TrainState.create(
            apply_fn=module.apply,
            params=params,
            tx=optimizer
        )
    summary=module.tabulate(jax.random.key(0), dummy_features)
    return state, summary