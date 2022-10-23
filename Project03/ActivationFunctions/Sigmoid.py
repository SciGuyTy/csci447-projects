from ActivationFunctions.ActivationFunction import ActivationFunction
import numpy as np

class Sigmoid(ActivationFunction):
    @classmethod
    def Sigmoid(action_potential: float) -> float:
        return np.vectorize(1 / (1 + np.exp(-action_potential)))

    # def __init__(self) -> None:
    #     super().__init__()

    # def compute(self, action_potential: float) -> float:
    #     return np.vectorize(1 / (1 + np.e(-action_potential)))