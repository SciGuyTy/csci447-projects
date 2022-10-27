from typing import List
import numpy as np

from ActivationFunctions.ActivationFunction import ActivationFunction


class Layer:
    def __init__(
        self,
        number_of_nodes: int,
        prev_layer,
        bias: float,
        activation: ActivationFunction,
    ) -> None:
        self.weights = np.random.rand(number_of_nodes, prev_layer)
        self.bias = np.repeat(np.array(bias), number_of_nodes)
        self.activation_function = activation
        self.number_of_nodes = number_of_nodes
        self.output = []


    def _compute_action_potential(self, input):
        if self.weights.size == 0:
            return input
        else:
            return np.matmul(self.weights, input) + self.bias

    def compute_output(self, input):
        action_potential = self._compute_action_potential(input)

        if self.activation_function:
            self.output = self.activation_function(action_potential)
        else:
            self.output = action_potential

    def update_weights(self):
        pass