import jax.numpy as jnp
import optax

from .components import *


class LeastSqRegressor(NeuralModel):
    
    def __init__(self, call_on_entry):
        super().__init__(call_on_entry,
                         lambda x, y: (x - y)**2 )


class SoftmaxClassifier(NeuralModel):
    
    def __init__(self, call_on_entry):
        super().__init__(call_on_entry,
                         optax.softmax_cross_entropy_with_integer_labels,
                         jnp.argmax)
        
        
class LossMonitor(MetricsMonitor):
    def __init__(self, model: NeuralModel,
                 *, early_stopping: Guard = None):
        
        metrics_fns = {"loss": model.loss_fn}
        super().__init__(metrics_fns, model, early_stopping=early_stopping)
        
        
class AccuracyMonitor(MetricsMonitor):
    
    def __init__(self, model: NeuralModel,
                 *, early_stopping: Guard = None):
        
        def accuracy(x, y):
            x = model.post_transform(x)
            return (x == y).astype(int)
        
        metrics_fns = {"accuracy": accuracy}
        super().__init__(metrics_fns, model, early_stopping=early_stopping)