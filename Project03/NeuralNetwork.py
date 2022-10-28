from typing import List
from Layer import Layer
import numpy as np
import pandas as pd
from ActivationFunctions.Sigmoid import Sigmoid

from Utilities.Preprocess import Preprocessor


class NeuralNetwork:
    def __init__(self, layers: List[Layer]) -> None:
        self.layers = layers

    def _feed_forward(self, input):
        layer_input = input

        # Compute the output of each layer using the output of the previous layer as input
        for layer in self.layers:
            layer.compute_output(layer_input)
            layer_input = layer.output

        # Return the output of the model
        return self.layers[-1].output

    def _back_propagate(self, learning_rate, momentum):
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

            curr_layer.weight_change += weight_change

    def train(
        self,
        training_data: pd.DataFrame,
        target_column: str,
        learning_rate: float,
        momentum: float,
        epochs: int,
        batch_size: int,
    ):
        training_columns = training_data.columns.drop(target_column)
        training_outputs = training_data[target_column].unique()

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
                    target = [0] * len(training_outputs)
                    target[pattern.loc[target_column] - 1] = 1

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

                    output_layer.weight_change += weight_change

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
    output_layer = Layer(4, 2, 0.0, Sigmoid())

    NN = NeuralNetwork([input_layer, hidden_layer1, output_layer])

    NN.train(PP.data, "class", 0.01, 0.1, 100, 10)

    test = [
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
    ]
    NN._feed_forward(test)
    print(NN.layers[-1].output)