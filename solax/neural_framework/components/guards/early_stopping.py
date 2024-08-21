from copy import copy
from flax.training.early_stopping import EarlyStopping

from .guard_base import *
from ..models import *


def get_flax_guard(params):
    return EarlyStopping(**params)


class EarlyStoppingGuard(Guard):
    
    def __init__(self,
                 model: NeuralModel,
                 *,
                 smaller_better: bool,
                 **early_stopping_params
                ):
        self._model = model
        self.to_watched = (lambda x: x) if smaller_better else (lambda x: -x)
        self.params = early_stopping_params
        self.reset()
        
        
    def reset(self):
        self._flax_guard = get_flax_guard(self.params)
        self._best_flax_state = self._model._flax_state

    
    def inform(self, data_entry):
        if len(data_entry) != 1:
            raise NotImplementedError("Early stopping implemented only for monitoring exactly one metrics.")
        val_orig = next(iter(data_entry.values()))
        val_watched = self.to_watched(val_orig)
        self._flax_guard = self._flax_guard.update(val_watched)
        if self._flax_guard.has_improved:
            self._best_flax_state = self._model._flax_state
        
        
    def __bool__(self):
        return self._flax_guard.should_stop
    
    
    @property
    def steps_from_best(self) -> int:
        return self._flax_guard.patience_count
    
    
    @property
    def best_model(self) -> NeuralModel:
        model_cpy = copy(self._model)
        model_cpy._flax_state = self._best_flax_state
        return model_cpy