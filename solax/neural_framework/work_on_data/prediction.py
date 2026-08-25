"""
Batched inference: applies a NeuralModel to a full dataset of features,
splitting it into batches to bound memory/compute per call.
"""
import numpy as np
from collections.abc import Sequence

from ..components import *


def predict_on_data(model: NeuralModel,
                    features: Sequence,
                    *,
                    batch_size: int | None = None
        ):
    """
    Runs "model" over the full "features" dataset, batch by batch, and
    concatenates the results back into one array.

    Input:

        - "model": an initialized NeuralModel (or subclass, e.g.
            SoftmaxClassifier).
        - "features": the full dataset of features to predict on.
        - "batch_size" (default=None): number of entries per batch;
            None means a single batch covering the whole dataset.

    Output:
        A NumPy array with "model"'s (post-transformed) predictions
        for every entry of "features", in the original order (no
        shuffling is used here).
    """
    @batchify(batch_sz=batch_size, shuffle=False)
    def predict(i, features):
        return model(features)

    full_out = np.concatenate([
        np.asarray(batch_out)
        for batch_out in predict(features)
    ])
    return full_out