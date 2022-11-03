from typing import Any
import numpy as np


class Layer:
    def __init__(
        self,
        num_nodes: int,
        prev_layer: Any = None,
        bias: np.array = None,
        activation: callable = None,
    ) -> None:
        self.num_nodes = num_nodes
        self.prev_layer = prev_layer

        # Shape should be num_nodes, np.zeros(num_nodes,)
        self.bias = bias
        self.activation = activation

        self.weights = self._generate_random_weights()
        self.output = None

        self.error = None
        self.delta = None
        
        if(prev_layer):
            self.weight_change = [np.zeros((prev_layer.num_nodes, num_nodes))]

        pass

    def _generate_random_weights(self):
        if self.prev_layer:
            return np.random.uniform(
                low=-0.1, high=0.1, size=(self.prev_layer.num_nodes, self.num_nodes)
            )
        else:
            return None

    def compute_output(self, input):
        # Compute action potential
        linear_combination = np.matmul(self.weights, input)
        
        if(self.bias):
            linear_combination += self.bias
        
        self.output = self.activation(linear_combination)

    # def update_weights(self):
    #     if(self.prev_layer):
    #         print(self.weight_change)

    #         self.weights = self.weights - sum(self.weight_change)
    #         self.weight_change = [np.zeros((self.prev_layer.num_nodes, self.num_nodes))]