import numpy as np

from Project04.NeuralNetwork import NeuralNetwork


class Utilities():
    @staticmethod
    def serialize_network(network: NeuralNetwork):
        chromosome = []

        # Flatten each layer weights matrix and concatenate them together
        for layer in network.layers:
            chromosome = np.concatenate((layer.weights.flatten(), chromosome))

        return chromosome

    @staticmethod
    def deserialize_network(network: NeuralNetwork, chromosome: np.array):
        layers = []
        curr_pointer = 0

        # Reconstruct each layer's weight matrix from the chromosome
        for layer in network.layers:
            # Get the shape of the layer
            shape = layer.weights.shape

            # The number of elements in the layer's weight matrix
            index_range = shape[0] * shape[1]

            # Parse weights from chromosome
            layer_weights = chromosome[curr_pointer: curr_pointer + index_range]

            # Construct array with rebuilt weight matrices
            layers.append(np.reshape(layer_weights, shape))
            curr_pointer += index_range

        return layers
