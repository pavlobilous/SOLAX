"""
neural_framework: a small, generic FLAX/JAX-based training layer used
internally by SOLAX to build and train neural networks (see SciPost
Phys. Codebases 51). It is not exposed at the general SOLAX interface
level; instead it supplies the machinery -- a model wrapper with
JIT-compiled train/predict/validate steps, batching, metrics
monitoring with optional early stopping, and Orbax-based checkpointing
-- behind the two ready-made, user-facing classes BasisClassifier and
BigBasisManager. The package is deliberately task-agnostic: it works
with any "features"/"labels" pair a user's architecture function
accepts, not just bit-encoded determinants.
"""
from .components import *
from .work_on_data import *
from .ready_classes import (
    LeastSqRegressor, SoftmaxClassifier,
    LossMonitor, AccuracyMonitor
)