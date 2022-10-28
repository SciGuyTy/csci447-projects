import numpy as np

class ActivationFunction:
    function = None
    delta = None

    def __init__(self) -> None:
        self.function = np.vectorize(self._function)
        self.delta = np.vectorize(self._delta)

    @staticmethod
    def _function() -> float:
        pass
    @staticmethod
    def _delta() -> float:
        pass
