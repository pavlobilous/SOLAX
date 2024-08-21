from collections.abc import Callable
import jax


def get_trainer(loss_fn: Callable):
    
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
    
    post_transform = jax.vmap(post_transform)

    @jax.jit
    def predictor(state, features):
        nn_out = state.apply_fn({'params': state.params}, features)
        return post_transform(nn_out)
    
    return predictor


def get_validator(metrics_fns: dict[str, Callable]):
    
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