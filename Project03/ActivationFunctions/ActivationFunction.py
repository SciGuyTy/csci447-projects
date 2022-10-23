import numpy as np

class ActivationFunction:
    def Sigmoid(action_potential: float) -> float:
        return np.vectorize(1 / (1 + np.exp(-action_potential)))