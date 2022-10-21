from typing import List
import numpy as np

NetworkShape = List[int]
class NeuralNetwork:

    RANDOM_WEIGHT_RANGE = (-.01, 0.01)

    def __init__(self, shape: NetworkShape):
        self.shape: NetworkShape = shape
        self.num_of_input_nodes: int = self.shape[0]
        self.num_of_output_nodes: int = self.shape[-1]
        self.hidden_layer_node_counts: List[int] = self.shape[1:-1]
        self.weight_matrices = self._construct_weight_matrices()

    def _construct_weight_matrices(self):
        weight_matrices = []

        # If there is no hidden layers
        if len(self.hidden_layer_node_counts) == 0:
            weight_matrix = self._construct_randomized_weight_matrix(self.num_of_input_nodes, self.num_of_output_nodes)
            weight_matrices.append(weight_matrix)
        else:
            # Add a weight matrix between the input layer and the first hidden layer
            weight_matrices.append(self._construct_randomized_weight_matrix(self.num_of_input_nodes, self.hidden_layer_node_counts[0]))

            # Add a weight matrix between each hidden layer
            for i in range(1, len(self.hidden_layer_node_counts)):
                weight_matrix = self._construct_randomized_weight_matrix(self.hidden_layer_node_counts[i-1], self.hidden_layer_node_counts[i])
                weight_matrices.append(weight_matrix)

            # Add a weight matrix between the last hidden layer and the output layer
            weight_matrices.append(self._construct_randomized_weight_matrix(self.hidden_layer_node_counts[-1], self.num_of_output_nodes))
        return weight_matrices

    def _construct_zeroed_weight_matrix(self, input_node_count: int, output_node_count: int):
        return np.zeros((output_node_count, input_node_count))

    def _construct_randomized_weight_matrix(self, input_node_count: int, output_node_count: int):
        rng = np.random.default_rng()
        # Construct a random array with values in the range of RANDOM_WEIGHT_RANGE
        return rng.random((output_node_count, input_node_count)) * (self.RANDOM_WEIGHT_RANGE[1]-self.RANDOM_WEIGHT_RANGE[0]) + self.RANDOM_WEIGHT_RANGE[0]