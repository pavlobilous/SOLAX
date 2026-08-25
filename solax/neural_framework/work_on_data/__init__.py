"""
Top-level batched training and prediction loops that operate a
NeuralModel over a full dataset: train_on_data (training with
optional validation and early stopping) and predict_on_data
(batched inference).
"""
from .prediction import predict_on_data
from .training import train_on_data