import numpy as np
import pandas as pd

from ActivationFunctions.Sigmoid import Sigmoid
from Layer import Layer
from typing import List, Callable

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
        # Construct the first layer
        layers = [Layer(shape[0], 0, 0, None, self.random_weight_range)]

        if len(hidden_layers) != 0:
            # Construct each of the hidden layers
            for i in range(1, len(shape)-1):
                layer = Layer(shape[i], shape[i-1], self.bias, Sigmoid(), self.random_weight_range)
                layers.append(layer)

        # Construct the last layer
        if self.regression:
            layers.append(Layer(shape[-1], shape[-2], self.bias, None, self.random_weight_range))
        else:
            layers.append(Layer(shape[-1], shape[-2], self.bias, Sigmoid(), self.random_weight_range))
        return layers

    # Predict the output using the output transformer
    def predict(self, input):
        return self.output_transformer(self._internal_predict(input))

    # Run a feed forward pass of the network on the given input
    def _internal_predict(self, input):
        output = input
        self.outputs = [output]

        # Loop through each layer
        for layer in self.layers[1:]:
            # Compute the action potential for each layer
            action_potential = layer._compute_action_potential(output).astype(float)

            # Run the output through the activation function for the layer
            if layer.activation is None:
                output = action_potential
            else:
                output = layer.activation.function(action_potential)
            # Store the output for use in backprop
            self.outputs.append(output)
        return output

    # Iteratively train the model
    def train(self, training_data: pd.DataFrame, target_column, learning_rate, momentum, iterations, batch_size, target_modifier):
        # Shuffle training data
        training_data = training_data.sample(frac=1)
        training_size = training_data.shape[0]

        # Loop for the given number of iterations
        for iteration in range(iterations):
            if iteration % 100 == 0:
                # Progress indicator
                print("Starting Epoch {}/{}".format(iteration+1, iterations))

            # Create each batch
            for i in range(int(training_size/batch_size)):
                prebatch = training_data[i*batch_size:i*batch_size+batch_size]
                if prebatch.shape[0] == 0:
                    continue
                # Create pairs of inputs and their targets for the batch
                inputs = prebatch.drop(columns=target_column)
                targets = [target_modifier(row) for _, row in prebatch.iterrows()]
                batch = zip(inputs.values, targets)

                # Run the backprop algorithm on the batches
                self.backprop_batch(batch, learning_rate, momentum)


    def backprop_batch(self, batch, learning_rate, momentum):
        weight_changes = []
        # Run the backprop algorithm and aggregate the weight changes
        for input, target in batch:
            weight_changes.append(self.backprop_once(learning_rate, input, target))

        if len(weight_changes) == 0:
            pass

        # Sum up the weight changes
        accumulated_weight_changes = weight_changes[0]
        for weights in weight_changes[1:]:
            for i in range(len(accumulated_weight_changes)):
                accumulated_weight_changes[i] = np.add(accumulated_weight_changes[i], weights[i])

        # Appy the weight changes
        for i in range(1, len(self.layers)):
            self.layers[i].weights -= (
                        accumulated_weight_changes[-i] + (np.multiply(momentum, self.layers[i].prev_weight_change)[0]))
            self.layers[i].prev_weight_change = accumulated_weight_changes[-i]

    def backprop_once(self, learning_rate, input, target):
        actual_output = self._internal_predict(input)

        weight_changes_by_layer = [] # Reversed (last layer at first index)
        dE_to_outputs = [] # [The dE_total for each of the node outputs in the previous layer]

        layer = self.layers[-1]

        # The partial derivatives of how much each of the final outputs contributes to the total error
        dE_total_to_output_j = - (target - actual_output)

        # The partial derivatives of how much each of the inputs to the final nodes contributes to the total error
        if self.regression:
            dE_total_to_input_j = dE_total_to_output_j
        else:
            dE_total_to_input_j = [actual_output[j] * (1-actual_output[j]) * dE_total_to_output_j[j] for j in range(len(dE_total_to_output_j))]

        # The partial derivatives of how much the output of each node in the previous layer contributes to the total error
        dE_total_to_output_prev_layer = layer.weights.T @ dE_total_to_input_j

        # Store the dE for each layer
        dE_to_outputs.append(dE_total_to_output_prev_layer)

        # Compute the partial derivatives for the last layer weights
        weights_partial_derivatives = np.outer(dE_total_to_input_j, self.outputs[-2])

        # Compute how much to change last layer weights by and save it
        weight_change = weights_partial_derivatives * learning_rate
        weight_changes_by_layer.append(weight_change)

        # Loop through the layers backwards starting at the second to last layer
        for i in range(-2, -len(self.layers), -1):
            layer = self.layers[i]
            output = self.outputs[i]

            # Calculate the partial derivative of the error to the input of layer
            dE_total_to_input_j = [output[j] * (1 - output[j]) * dE_to_outputs[-i - 2][j]
                                    for j in range(len(output))]

            # Calculate the partial derivative of the error to the output of the layer before
            dE_total_to_output_prev_layer = layer.weights.T @ dE_total_to_input_j
            dE_to_outputs.append(dE_total_to_output_prev_layer)

            # Compute the partial derivatives for the layer's weights
            weights_partial_derivatives = np.outer(dE_total_to_input_j, self.outputs[i-1])

            # Compute how much to change the layer's weights by and save it
            weight_change = weights_partial_derivatives * learning_rate
            weight_changes_by_layer.append(weight_change)

        # Return the output for each layer
        return weight_changes_by_layer

    # Calculate the actual error (used in testing)
    def error(self, input, target):
        output = self._internal_predict(input)

        total = 0
        for i in range(len(output)):
            total += (output[i] - target[i])**2
        return total / 2