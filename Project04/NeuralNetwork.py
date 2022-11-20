import numpy as np
import pandas as pd


from typing import List, Callable

from Project04.ActivationFunctions.Sigmoid import Sigmoid
from Project04.Layer import Layer


class NeuralNetwork:
    def __init__(self, shape: List[int], output_transformer: Callable, regression: bool, random_weight_range) -> None:
        self.output_transformer = output_transformer
        self.bias = 0
        self.regression = regression
        self.random_weight_range = random_weight_range
        self.layers = self.construct_layers(shape)
        self.action_potentials = []
        self.outputs = []

    def construct_layers(self, shape):
        hidden_layers = shape[1:-1]

        layers = [Layer(shape[0], 0, 0, None, self.random_weight_range)]
        if len(hidden_layers) != 0:
            for i in range(1, len(shape)-1):
                layer = Layer(shape[i], shape[i-1], self.bias, Sigmoid(), self.random_weight_range)
                layers.append(layer)
        if self.regression:
            layers.append(Layer(shape[-1], shape[-2], self.bias, None, self.random_weight_range))
        else:
            layers.append(Layer(shape[-1], shape[-2], self.bias, Sigmoid(), self.random_weight_range))
        return layers

    def predict(self, input):
        return self.output_transformer(self._internal_predict(input))

    def _internal_predict(self, input):
        output = input
        self.outputs = [output]
        for layer in self.layers[1:]:
            action_potential = layer._compute_action_potential(output).astype(float)
            if layer.activation is None:
                output = action_potential
            else:
                output = layer.activation.function(action_potential)
            self.outputs.append(output)
        return output

    def train(self, training_data: pd.DataFrame, target_column, learning_rate, momentum, iterations, batch_size, target_modifier):
        # shuffle training data
        training_data = training_data.sample(frac=1)
        training_size = training_data.shape[0]
        for iteration in range(iterations):
            if iteration % 100 == 0:
                # print("Starting Epoch {}/{}".format(iteration+1, iterations))
                pass
            for i in range(int(training_size/batch_size)):
                prebatch = training_data[i*batch_size:i*batch_size+batch_size]
                if prebatch.shape[0] == 0:
                    continue
                inputs = prebatch.drop(columns=target_column)
                targets = [target_modifier(row) for _, row in prebatch.iterrows()]
                batch = zip(inputs.values, targets)

                self.backprop_batch(batch, learning_rate, momentum)




    def error(self, input, target):
        output = self._internal_predict(input)

        total = 0
        for i in range(len(output)):
            total += (output[i] - target[i])**2
        return total / 2