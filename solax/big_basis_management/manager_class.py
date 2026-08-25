"""
BigBasisManager: orchestrates the NN-assisted basis-optimization
procedure of SciPost Phys. Codebases 51 Sec. 3 for a single big basis of
"candidates" determinants -- random selection, a coefficient-magnitude
cutoff, and training/applying a BasisClassifier to predict which
candidates are important.
"""
import numpy as np
import jax
from dataclasses import dataclass

from solax.quantum_core import *
from solax.neural_framework import *
from .basis_classifier import *
from .training_defaults import *



@dataclass
class BigBasisManager:
    """
    Drives the three-step NN-assisted basis-optimization procedure of
    SciPost Phys. Codebases 51 Sec. 3 (Fig. "Neural network support for
    tackling big basis sets") for one big basis of not-yet-classified
    "candidates" determinants ("big_basis"):
      (A) sample_subbasis: randomly draw a small subset of candidates.
      (B) derive_abs_coeff_cut: after diagonalizing on that random
          selection, derive a coefficient-magnitude cutoff calibrated
          to select roughly a target number of "important" determinants.
      (C) train_classifier / predict_impt_subbasis: train "classifier"
          (a BasisClassifier) on the random selection, labeled by that
          cutoff, then run it over the full "candidates" basis to
          predict which candidates are important.

    Each BigBasisManager instance is bound to the particular "big_basis"
    it was constructed with; a new BigBasisManager must be created for
    each new big basis of candidates. The "classifier" it holds, by
    contrast, can be reused across BigBasisManager instances -- and is
    itself updated in place by train_classifier when early stopping
    picks an earlier snapshot -- "transferring in this way the NN
    experience from case to case."
    """
    big_basis: Basis
    classifier: BasisClassifier


    def sample_subbasis(self, key, random_num: int
                       ) -> Basis:
        """
        Step (A) of the procedure: randomly draws "random_num"
        determinants (without replacement) from "big_basis", using
        "key" (a JAX PRNG key) to shuffle candidate positions. Returns
        the random selection as a new Basis; "big_basis" itself is
        left unchanged.
        """
        inds = shuffled_inds(key, length=len(self.big_basis))[:random_num]
        return self.big_basis[inds]


    def derive_abs_coeff_cut(self, target_num: int, rand_substate: State
                            ) -> float:
        """
        Step (B) of the procedure: given "rand_substate" (a State
        obtained by diagonalizing on a random selection of candidates,
        e.g. via sample_subbasis), derives a cutoff on a determinant's
        "weight" -- the absolute value of its expansion coefficient,
        per SciPost Phys. Codebases 51 Sec. 3 -- chosen so that the
        fraction of "rand_substate" determinants with weight >=
        abs_coeff_cut equals target_num / len(big_basis). Assuming
        "rand_substate" is representative of the full "big_basis"
        candidate pool, applying this same cutoff there is expected to
        select approximately "target_num" important determinants out
        of the whole pool. The returned cutoff is the midpoint between
        the two sorted |coefficient| values straddling that fraction.
        """
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
                         early_stop_params: dict = None,
                         **train_kwargs
                        ):
        """
        Step (C) of the procedure (training half): trains "classifier"
        as a binary important/unimportant classifier on "train_state"
        -- typically the same random selection that was diagonalized
        to derive "abs_coeff_cut" (see sample_subbasis /
        derive_abs_coeff_cut). Labels are obtained by thresholding each
        determinant's weight (|coefficient|) in "train_state" against
        "abs_coeff_cut"; features are the bit-encoded determinants of
        train_state.basis. "key" is split to both shuffle "train_state"
        into a training/validation split (a val_frac fraction held out
        for validation) and to drive train_on_data itself.

        "batch_size" and "epochs" are required; "val_frac" and the
        other batching/reporting knobs default to DEFAULT_TRAIN_KWARGS
        (see training_defaults.py) and, together with any further
        keywords accepted by train_on_data, can be overridden via
        **train_kwargs.

        Unless **train_kwargs already supplies a "val_metrics" monitor,
        validation is tracked with a fresh AccuracyMonitor; in that
        default case, if "early_stop" is True, the monitor is further
        given an EarlyStoppingGuard (configured via "early_stop_params",
        forwarded to Flax's own EarlyStopping) that snapshots
        "classifier" whenever validation accuracy improves. If early
        stopping triggers, "classifier" is replaced in place by that
        best snapshot.

        Returns True if early stopping triggered (ending training
        before all "epochs" completed), False otherwise.
        """
        train_kwargs = (
            DEFAULT_TRAIN_KWARGS
            | dict(batch_size=batch_size, epochs=epochs)
            | train_kwargs
        )
        early_stop_params = early_stop_params or {}
        
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
        """
        Step (C) of the procedure (inference half): runs "classifier"
        over every determinant of "big_basis" (including the training
        subset, e.g. from sample_subbasis) and returns the sub-Basis of
        "big_basis" it predicts to be important. Since the training
        subset is itself part of "big_basis", callers typically strip
        it back out of the result afterward (e.g. via
        "% state_train.basis").
        """
        impt01 = predict_on_data(self.classifier, self.big_basis._encoding,
                        batch_size=batch_size)
        return self.big_basis[impt01.astype(bool)]