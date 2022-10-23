from audioop import reverse
from typing import List
from Layer import Layer
import numpy as np
import pandas as pd


class NeuralNetwork:
    def __init__(self, layers: List[Layer], input: np.array, target) -> None:
        self.layers = layers
        self.input = input
        self.target = target

    def _feed_forward(self):
        layer_input = self.input

        # Compute the output of each layer using the output of the previous layer as input
        for layer in self.layers:
            layer.compute_output(layer_input)
            layer_input = layer.output

        # Return the output of the model
        return self.layers[-1].output

    def _compute_delta(self, observed, expected):
        delta = np.subtract(expected, observed)
        return np.multiply(np.multiply(delta, observed), np.subtract(1, observed))

    def _compute_delta_h(self, observed, expected):
        delta = np.subtract(expected, observed)
        return np.multiply(delta, observed)

    def _back_propagate(self, learning_rate):
        for index in range(len(self.layers) - 2, 1, -1):
            curr_layer = self.layers[index - 1]
            prev_layer = self.layers[index]

            print(curr_layer.number_of_nodes, prev_layer.number_of_nodes)
        
            weighted_sum = np.matmul(prev_layer.output.T, prev_layer.weights)

            # Compute error for layer
            layer_err = np.multiply(
                np.vectorize(self._compute_delta_h)(curr_layer.output, self.target), weighted_sum
            )

            # Update weights
            weight_change = learning_rate * np.multiply(layer_err, curr_layer.output)

            self.layers[index - 1].weights = curr_layer.weights + weight_change

    def train(
        self,
        # training_data: pd.DataFrame,
        # tuning_data: pd.DataFrame,
        learning_rate: float,
    ):
        for i in range(100):
            output = self._feed_forward()
            output_err = np.vectorize(self._compute_delta)(output, self.target)

            self._feed_forward()
            self._back_propagate(learning_rate)

            weight_change = learning_rate * np.outer(
                output_err, self.layers[-2].output
            )
            self.layers[-1].weights = self.layers[-1].weights + weight_change

        print(self.layers[-1].output)


def Sigmoid(action_potential: float) -> float:
    return 1 / (1 + np.exp(-action_potential))


if __name__ == "__main__":
    input_layer = Layer(2, 0, 0.0, None)
    hidden_layer = Layer(4, 2, 0.0, np.vectorize(Sigmoid))
    output_layer = Layer(3, 4, 0.0, np.vectorize(Sigmoid))

    NN = NeuralNetwork([input_layer, hidden_layer, output_layer], [0.05, 0.10], 0.4)

    # print(NN._feed_forward())
    # print(NN._compute_delta(1, 1.5))
    # for i in range(2):
    #     NN._back_propagate(0.5)

    NN.train(0.5)
