from typing import List
from ActivationFunctions.ActivationFunction import ActivationFunction
import numpy as np


class Softmax(ActivationFunction):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def _function(vector: List[float]) -> float:
        softmax_vector = np.exp(vector) / np.exp(vector).sum()
        bounded = ([0] * len(vector))
        bounded[np.argmax(softmax_vector)] = 1
        return bounded

    @staticmethod
    def _delta(observed: float) -> float:
        return np.multiply(observed, np.subtract(1, observed))
