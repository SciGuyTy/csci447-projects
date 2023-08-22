import numpy as np

from Project04.ActivationFunctions.ActivationFunction import ActivationFunction

class Sigmoid(ActivationFunction):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def _function(action_potential) -> float:
        return 1 / (1 + np.exp(-action_potential))

    @staticmethod
    def _delta(observed: float) -> float:
        return np.multiply(observed, np.subtract(1, observed))
