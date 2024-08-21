from collections.abc import Callable
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState


class Module(nn.Module):
    call_on_entry: Callable
    
    @nn.compact
    def __call__(self, features):
        call_on_data = jax.vmap(self.call_on_entry)
        return call_on_data(features)
    
    
def create_state_and_summary(key, call_on_entry, dummy_features, optimizer):
    module = Module(call_on_entry)
    params = module.init(key, dummy_features)['params']
    state = TrainState.create(
            apply_fn=module.apply,
            params=params,
            tx=optimizer
        )
    summary=module.tabulate(jax.random.key(0), dummy_features)
    return state, summary