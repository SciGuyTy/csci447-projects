import numpy as np




class Layer:
    RANDOM_WEIGHT_RANGE = (-1, 1)

    def __init__(self, number_of_nodes, prev_layer_nodes, bias, activation, random_weight_range):
        self.RANDOM_WEIGHT_RANGE = random_weight_range
        self.number_of_nodes = number_of_nodes
        self.bias = np.repeat(np.array(bias), number_of_nodes)
        self.activation = activation
        self.weights = self._construct_randomized_weight_matrix(prev_layer_nodes, number_of_nodes)
        self.output = None
        self.prev_weight_change = [np.zeros((number_of_nodes, prev_layer_nodes))]

    def _construct_randomized_weight_matrix(self, input_node_count: int, output_node_count: int):
        rng = np.random.default_rng()
        # Construct a random array with values in the range of RANDOM_WEIGHT_RANGE
        return rng.random((output_node_count, input_node_count)) * (
                    self.RANDOM_WEIGHT_RANGE[1] - self.RANDOM_WEIGHT_RANGE[0]) + self.RANDOM_WEIGHT_RANGE[0]

    def _compute_action_potential(self, input):
        # Compute the action potential for the layer
        return (self.weights @ input) + self.bias