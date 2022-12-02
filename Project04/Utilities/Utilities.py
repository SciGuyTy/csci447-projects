import numpy as np
import pandas as pd

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

        return network

    @classmethod
    def normalize_set_by_params(cls, data: pd.DataFrame, norm_params: pd.DataFrame):
        """
        Perform min-max normalization on a group of instances belonging to the dataset

        Parameters
        -----------
        data
            The data with which to normalize

        """
        for index, row in norm_params.iterrows():
            feature = row["feature"]

            # Cast dtype of features to be normalized as a float
            data[feature] = data[feature].astype("float64")

            # Retrieve the minimum and maximum values for the given column
            minimum_value = row['min']
            maximum_value = row['max']

            # Define the newly normalized feature in the dataset, and assign its values to
            # the normalized version of the original data

            if maximum_value - minimum_value == 0:
                data[feature] = 0
            else:
                data[feature] = (data[feature] - minimum_value) / (
                        maximum_value - minimum_value
                )

        # Return the normalized data
        return data
