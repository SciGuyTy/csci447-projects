from typing import List
import numpy as np
import pandas as pd

from Project03.ActivationFunctions.ActivationFunction import ActivationFunction



class Layer:

    RANDOM_WEIGHT_RANGE = (-.01, 0.01)

    def __init__(
        self,
        number_of_nodes: int,
        prev_layer,
        bias: float,
        activation: ActivationFunction,
    ) -> None:
        self.weight_change = [np.zeros((number_of_nodes, prev_layer))]
        #self.weights = np.random.rand(number_of_nodes, prev_layer)
        self.weights = self._construct_randomized_weight_matrix(prev_layer, number_of_nodes)
        self.bias = np.repeat(np.array(bias), number_of_nodes)
        self.activation_function = activation
        self.number_of_nodes = number_of_nodes
        self.prev_layer = prev_layer
        self.output = []
        self.error_signal = []

    def _construct_randomized_weight_matrix(self, input_node_count: int, output_node_count: int):
        rng = np.random.default_rng()
        # Construct a random array with values in the range of RANDOM_WEIGHT_RANGE
        return rng.random((output_node_count, input_node_count)) * (self.RANDOM_WEIGHT_RANGE[1]-self.RANDOM_WEIGHT_RANGE[0]) + self.RANDOM_WEIGHT_RANGE[0]

    def _compute_action_potential(self, input: List[float]):
        """
        Compute the action potential matrix for the layer (sum of each weight multiplied
        by the previous layer's output for the corresponding node)
        
        Parameters
        ----------
        input: List[float]
            The output vector from the previous layer
        """

        if self.weights.size == 0:
            return input
        else:
            return np.matmul(self.weights, input) + self.bias

    def compute_output(self, input: List[float]):
        """
        Compute the output for the layer. If an activation is provided,
        this is the result of passing the action potential through
        the activation function
        
        Parameters
        ----------
        input: List[float]
            The output vector from the previous layer
        """
        action_potential = self._compute_action_potential(input).astype(float)

        if self.activation_function:
            self.output = self.activation_function.function(action_potential)
        else:
            self.output = action_potential

    def update_weights(self):
        """
        Update the weight matrix for this layer
        """
        self.weights = self.weights + sum(self.weight_change)
        self.weight_change = [np.zeros((self.number_of_nodes, self.prev_layer))]

    def compute_delta(self, target: List[float]=None):
        """
        Compute the derivative of the activation function for the layer. 
        If an activation is not provided, simply return 1.0 (derivative of 
        constant)
        
        Parameters
        ----------
        target: List[float]
            The target/expected output (only applicable for the output layer)
        """

        # Handle computing derivative for layer
        if type(target) == list:
            loss = np.subtract(target, self.output)
            if self.activation_function:
                return np.multiply(loss, self.activation_function.delta(self.output))
            else:
                return loss
        else:
            loss = np.subtract(np.subtract(1, self.output), self.output)
            return np.multiply(loss, self.output)
