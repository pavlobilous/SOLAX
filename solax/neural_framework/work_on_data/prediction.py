import numpy as np
from collections.abc import Sequence

from ..components import *


def predict_on_data(model: NeuralModel,
                    features: Sequence,
                    *,
                    batch_size: int | None = None
        ):
    
    @batchify(batch_sz=batch_size, shuffle=False)
    def predict(i, features):
        return model(features)
    
    full_out = np.concatenate([
        np.asarray(batch_out)
        for batch_out in predict(features)
    ])
    return full_out