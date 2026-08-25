"""
Factories building the three JIT-compiled, vmapped core functions
driving a NeuralModel: one gradient-descent training step
(get_trainer), one forward-pass prediction (get_predictor), and one
metrics evaluation (get_validator). Each factory takes a per-entry
(non-vectorized) callable and returns a jax.jit-compiled function that
operates on a full batch, vmapping the callable over the batch
dimension and, where relevant, batch-averaging its output.
"""
from collections.abc import Callable
import jax


def get_trainer(loss_fn: Callable):
    """
    Builds the JIT-compiled training-step function for a Flax
    TrainState. "loss_fn(nn_out, label) -> loss" is a per-entry loss
    function; it is vmapped over the batch and its output
    batch-averaged to obtain the scalar loss that is differentiated.

    Returns a function trainer(state, features, labels) -> new_state
    that computes the gradient of the (mean) loss w.r.t. "state"'s
    params on the given batch and returns the state after one
    optimizer update (state.apply_gradients); "state" itself is not
    mutated.
    """
    loss_fn = jax.vmap(loss_fn)

    @jax.jit
    def trainer(state, features, labels):
        def loss_from_params(params):
            nn_out = state.apply_fn({'params': params}, features)
            return loss_fn(nn_out, labels).mean()

        grad_fn = jax.grad(loss_from_params)
        grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state

    return trainer


def get_predictor(post_transform: Callable):
    """
    Builds the JIT-compiled prediction function for a Flax TrainState.
    "post_transform(nn_out) -> nn_out_transformed" is a per-entry
    post-processing function (e.g. argmax for a classifier, or the
    identity for a regressor); it is vmapped over the batch.

    Returns a function predictor(state, features) -> predictions that
    runs the forward pass with "state"'s current params and applies
    "post_transform" to each entry of the output.
    """
    post_transform = jax.vmap(post_transform)

    @jax.jit
    def predictor(state, features):
        nn_out = state.apply_fn({'params': state.params}, features)
        return post_transform(nn_out)

    return predictor


def get_validator(metrics_fns: dict[str, Callable]):
    """
    Builds the JIT-compiled metrics-evaluation function for a Flax
    TrainState. "metrics_fns" maps each metric name to a per-entry
    "metrics_fn(nn_out, label) -> value" function; each is vmapped
    over the batch and its output batch-averaged.

    Returns a function validator(state, features, labels) -> dict
    mapping each metric name to its batch-averaged value, computed
    from a single forward pass with "state"'s current params.
    """
    metrics_fns = {
        nm : jax.vmap(fn)
        for nm, fn in metrics_fns.items()
    }

    @jax.jit
    def validator(state, features, labels):
        nn_out = state.apply_fn({'params': state.params}, features)
        mvals = {
            nm : fn(nn_out, labels).mean()
            for nm, fn in metrics_fns.items()
        }
        return mvals

    return validator