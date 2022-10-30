from typing import List, Callable
from Project03.Layer import Layer
import numpy as np
import pandas as pd
from Project03.ActivationFunctions.Sigmoid import Sigmoid
from Project03.ActivationFunctions.Softmax import Softmax

from Project03.Utilities.Preprocess import Preprocessor


class NeuralNetwork:
    def __init__(self, shape: List[int], output_transformer: Callable, regression: bool) -> None:
        self.output_transformer = output_transformer
        self.bias = 0
        self.regression = regression
        self.layers = self.construct_layers(shape)

    def construct_layers(self, shape):
        hidden_layers = shape[1:-1]

        layers = [Layer(shape[0], 0, 0, None)]
        if len(hidden_layers) != 0:
            for i in range(1, len(shape)-1):
                layer = Layer(shape[i], shape[i-1], self.bias, Sigmoid())
                layers.append(layer)
        if self.regression:
            layers.append(Layer(shape[-1], shape[-2], self.bias, None))
        else:
            layers.append(Layer(shape[-1], shape[-2], self.bias, Softmax()))
        return layers

    def predict(self, input: List[float]):
        """
        Perform a feed forward pass on the network for a given input
        and transform it according to the output_transformer

        Parameters
        ----------
        input: List[float]
            The input vector to pass through the network
        """
        return self.output_transformer(self._feed_forward(input))

    def _feed_forward(self, input: List[float]):
        """
        Perform a feed forward pass on the network for a given input

        Parameters
        ----------
        input: List[float] 
            The input vector to pass through the network
        """

        # Compute the output of each layer using the output of the previous layer as input
        for layer in self.layers:
            layer.compute_output(input)
            input = layer.output

        # Return the output of the model
        return np.array(self.layers[-1].output)

    def _back_propagate(self, learning_rate: float, momentum: float):
        """
        Perform backpropagation on the network

        Parameters
        ----------
        learning_rate: float
            The learning rate used to guide the modify the 'strength' of
            the change in weights for each backprop pass

        momentum: float
            The momentum used to modify the 'resiliance to change' of the 
            change in weights for each backprop pass    

        """

        for index in range(len(self.layers) - 2, 0, -1):
            # Get reference to current, preceding, and following layer
            curr_layer = self.layers[index]
            prev_layer = self.layers[index + 1]
            next_layer = self.layers[index - 1]

            # Compute weighted error
            weighted_sum = (
                prev_layer.weights * prev_layer.error_signal[:, np.newaxis]
            ).sum(axis=0)

            # Compute error for layer
            curr_layer.error_signal = np.multiply(
                curr_layer.compute_delta(),
                weighted_sum,
            )

            # Construct weight update matrix
            weight_change = learning_rate * np.tile(
                next_layer.output, (curr_layer.error_signal.size, 1)
            ) * curr_layer.error_signal[:, np.newaxis] + (
                curr_layer.weight_change[-1] * momentum
            )

            curr_layer.weight_change.append(weight_change)

    def train(
        self,
        training_data: pd.DataFrame,
        target_column: str,
        learning_rate: float,
        momentum: float,
        epochs: int,
        batch_size: int,
        target_modifier: callable = lambda x: x.to_numpy()
    ):
        """
        Train the network using backpropagation

        Parameters
        ----------
        training_data: pd.DataFrame
            The data with which to train the model on
        
        target_column: str
            The label for the output column in the data

        learning_rate: float
            The learning rate used to guide the modify the 'strength' of
            the change in weights for each backprop pass

        momentum: float
            The momentum used to modify the 'resiliance to change' of the 
            change in weights for each backprop pass

        epochs: int
            The number of epochs to perform (number of times to run backprop
            on the entire training data set)

        batch_size: int
            The size of each training batch

        target_modifier: callable
            A function that modifies the target pattern. Defaults to an identity
            function that simply returns the input pattern
        """

        training_columns = training_data.columns.drop(target_column)

        # Split training data into batches
        num_batches = len(training_data.index) / batch_size
        batched_data = np.array_split(training_data.sample(frac=1), num_batches)

        for epoch in range(epochs):
            # Run backpropagation with batches
            for batch in batched_data:
                # Run backpropagation on the batch
                for sample in batch.iterrows():
                    _, pattern = sample

                    # Construct target vector
                    target = target_modifier(pattern)

                    # Reference to the output layer of the network
                    output_layer = self.layers[-1]

                    # Feed the pattern through the network
                    self._feed_forward(pattern[training_columns].to_numpy())

                    # Compute the error signal for the output layer
                    output_layer.error_signal = output_layer.compute_delta(target)

                    # Record the weight change for the output layer
                    weight_change = learning_rate * np.outer(
                        output_layer.error_signal, self.layers[-2].output
                    )

                    output_layer.weight_change.append(weight_change)

                    # Perform back propagation on any hidden layers
                    if len(self.layers) > 2:
                        self._back_propagate(learning_rate, momentum)


                # Apply the weight change for each layer
                for layer in self.layers:
                    layer.update_weights()

            print(f"Finished Epoch: {epoch}")

if __name__ == "__main__":
    file_path = "../datasets/classification/Soybean/soybean-small.data"
    column_names = [
        "date",
        "plant_stand",
        "precip",
        "temp",
        "hail",
        "crop_hist",
        "area_damaged",
        "severity",
        "seed_tmt",
        "germination",
        "plant_growth",
        "leaves",
        "leafspots_halo",
        "leafspots_marg",
        "leafspot_size",
        "leaf_shread",
        "leaf_malf",
        "leaf_mild",
        "stem",
        "lodging",
        "stem_cankers",
        "canker_lesion",
        "fruiting_bodies",
        "external_decay",
        "mycelium",
        "int_discolor",
        "sclerotia",
        "fruit_pods",
        "fruit_spots",
        "seed",
        "mold_growth",
        "seed_discolor",
        "seed_size",
        "shriveling",
        "roots",
        "class",
    ]

    converters = {
        "class": lambda x: int(x[1])
    }  # Convert the class column from ints to booleans

    # Process the data
    PP = Preprocessor()
    PP.load_raw_data_from_file(file_path, column_names, converters=converters)

    input_layer = Layer(35, 0, 0.0, None)
    hidden_layer1 = Layer(2, 35, 0.0, Sigmoid())
    output_layer = Layer(4, 2, 0.0, Softmax())


    def output_transformer(output_vector: np.array):
        return output_vector.argmax() + 1

    NN = NeuralNetwork([35, 2, 4], output_transformer, False)

    classes = PP.data["class"].unique()

    def classification_modifier(pattern: pd.Series):
        target = [0] * len(classes)
        target[pattern.loc["class"] - 1] = 1
        return target

    NN.train(PP.data, "class", 0.01, 0.1, 100, 10, classification_modifier)

    test = np.array([
        0,
        1,
        2,
        1,
        0,
        3,
        1,
        1,
        0,
        2,
        1,
        1,
        0,
        2,
        2,
        0,
        0,
        0,
        1,
        0,
        1,
        2,
        0,
        0,
        0,
        0,
        0,
        3,
        4,
        0,
        0,
        0,
        0,
        0,
        1,
    ])
    print(NN._feed_forward(test))
    print(NN.layers[-1].output)