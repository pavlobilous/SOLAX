"""
The NeuralModel wrapper around a Flax module, plus its supporting
pieces: flax_fundament (builds the Flax Module/TrainState) and
orbax_save_load (Orbax-based checkpointing of the Flax state).
"""
from .model_class import NeuralModel