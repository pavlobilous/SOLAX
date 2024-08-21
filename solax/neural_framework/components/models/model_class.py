import os
from collections.abc import Callable

from .flax_fundament import *
from ..jitted_core import *
from .orbax_save_load import *
    
    
class NeuralModel:
    """
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
        self.call_on_entry = call_on_entry
        self.loss_fn = loss_fn
        self.post_transform = post_transform
        self._trainer = get_trainer(loss_fn)
        self._predictor = get_predictor(post_transform)
        self._flax_state = None
             
            
    def initialize(self, key, dummy_features, optimizer):
        self._flax_state, self._summary = create_state_and_summary(
            key, self.call_on_entry, dummy_features, optimizer
        )


    def __call__(self, features):
        return self._predictor(self._flax_state, features)
    
    
    def train(self, features, labels):
        self._flax_state = self._trainer(self._flax_state, features, labels)
        
   
    def print_summary(self):
        print(self._summary)
        
        
    def save_state(self, fld: str):
        if self._flax_state is not None:
            save_flax_state(fld, self._flax_state)
        else:
            raise IOError('Cannot save state of an unitialized NeuralModel.')
        
        
    def load_state(self, fld: str):
        if not os.path.exists(fld):
            raise FileNotFoundError(f"Cannot load from here. Path does not exist.")
        if self._flax_state is not None:
            self._flax_state = load_flax_state(fld, self._flax_state)
        else:
            raise IOError('Cannot load state for an unitialized NeuralModel. Call "initialize" first and try again.')