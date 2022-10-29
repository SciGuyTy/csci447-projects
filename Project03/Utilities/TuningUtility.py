import math

import pandas as pd
from typing import Callable
from Project03.Algorithm.NeuralNetwork import NeuralNetwork
class TuningUtility:

    def __init__(self, folds, tuning_data, num_inputs, num_outputs, cost_function: Callable):
        self.folds = folds
        self.tuning_data = tuning_data
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.cost_function = cost_function

        self.learning_rate = 1
        self.momentum = 0

        # Use 10 minibatches
        self.minibatch_size = math.ceil(len(folds[0]) * 0.1)

        self.iterations = 20

    def tune_for_h_hidden_layers(self, h):
        shapes = self._get_all_shapes_for_hidden_layers(h)
        best_models_for_layers = []
        for training_data, hold_out in self.folds:
            cost_for_shapes = dict()
            for shape in shapes:
                nn = NeuralNetwork(shape)
                nn.train(training_data, self.tuning_data, self.learning_rate, self.cost_function, self.momentum, self.iterations, self.minibatch_size)

                # TODO Calculate performance
                cost = 0

                cost_for_shapes[shape] = (cost, nn, shape)
            best_models_for_layers.append(min(cost_for_shapes.values()))
        return best_models_for_layers

    def _get_all_shapes_for_hidden_layers(self, num_hidden_layers):
        shapes = self._generate_possible_configs(self.num_inputs, self.num_outputs, num_hidden_layers)
        for i in range(len(shapes)):
            shapes[i] = [self.num_inputs] + shapes[i] + [self.num_outputs]

        return shapes

    def _generate_possible_configs(self, num_of_inputs, num_of_outputs, num_of_hidden_layers):
        if num_of_hidden_layers == 1:
            return [[i] for i in range(num_of_outputs, num_of_inputs + 1)]
        output = []
        for i in range(num_of_outputs, num_of_inputs + 1):
            x = self.generate_possible_configs(i, num_of_outputs, num_of_hidden_layers - 1)
            for j in x:
                output.append([i] + j)
        return output
