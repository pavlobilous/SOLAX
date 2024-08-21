import numpy as np
import jax
from dataclasses import dataclass

from solax.quantum_core import *
from solax.neural_framework import *
from .basis_classifier import *
from .training_defaults import *



@dataclass
class BigBasisManager:
    big_basis: Basis
    classifier: BasisClassifier
        
        
    def sample_subbasis(self, key, random_num: int
                       ) -> Basis:
        inds = shuffled_inds(key, length=len(self.big_basis))[:random_num]
        return self.big_basis[inds]
    
    
    def derive_abs_coeff_cut(self, target_num: int, rand_substate: State
                            ) -> float:
        abs_coeff_srt = np.sort(
            np.abs(rand_substate.coeffs)
        )[::-1]
        impt_frac = target_num / len(self.big_basis)
        impt_num = int(impt_frac * len(abs_coeff_srt))
        abs_coeff_cut = \
            (abs_coeff_srt[impt_num - 1] + abs_coeff_srt[impt_num]) / 2   
        return abs_coeff_cut
 
   
    def train_classifier(self, key, train_state: State, abs_coeff_cut: float,
                         *,
                         batch_size: int,
                         epochs: int,
                         early_stop: bool,
                         early_stop_params: dict = {},
                         **train_kwargs
                        ):
        train_kwargs = (
            DEFAULT_TRAIN_KWARGS
            | dict(batch_size=batch_size, epochs=epochs)
            | train_kwargs
        )
        
        impt01 = (np.abs(train_state.coeffs) >= abs_coeff_cut).astype(np.int8)
        
        val_frac = train_kwargs.pop("val_frac")
        
        key, subkey = jax.random.split(key)
        train_inds, val_inds = np.split(
            shuffled_inds(subkey, length=len(train_state)),
            [int(len(train_state) * (1 - val_frac))]
        )
        data_dict = dict(
            train_data=(train_state.basis._encoding[train_inds], impt01[train_inds]),
            val_data=(train_state.basis._encoding[val_inds], impt01[val_inds])    
        )
        
        val_metrics = train_kwargs.pop("val_metrics", None)
        
        if val_metrics is None:
            if early_stop:
                es = EarlyStoppingGuard(self.classifier,
                                        smaller_better=False,
                                        **early_stop_params)
            else:
                es = None
            val_metrics = AccuracyMonitor(self.classifier, early_stopping=es)
        
        key, subkey = jax.random.split(key)
        early_stopped = train_on_data(subkey, self.classifier,
                                      **data_dict,
                                      val_metrics=val_metrics,
                                      **train_kwargs)
        if early_stopped:
            self.classifier = val_metrics.early_stopping.best_model
        
        return early_stopped
    
    
    def predict_impt_subbasis(self, *, batch_size):
        impt01 = predict_on_data(self.classifier, self.big_basis._encoding,
                        batch_size=batch_size)
        return self.big_basis[impt01.astype(bool)]