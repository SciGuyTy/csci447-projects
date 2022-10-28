from typing import List
from Layer import Layer
import numpy as np
import pandas as pd

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

    def _compute_delta(self, observed, expected):
        delta = np.subtract(expected, observed)
        return np.multiply(np.multiply(delta, observed), np.subtract(1, observed))

    def _compute_delta_h(self, observed, expected):
        delta = np.subtract(expected, observed)
        return np.multiply(delta, observed)

    def _back_propagate(self, learning_rate, momentum):
        for index in range(len(self.layers) - 2, 0, -1):
            curr_layer = self.layers[index]
            prev_layer = self.layers[index + 1]
            next_layer = self.layers[index - 1]

            # Compute weighted error
            weighted_sum = (prev_layer.weights * prev_layer.error_signal[:, np.newaxis]).sum(axis=0) 

            # Compute error for layer
            layer_err = np.multiply(
                np.vectorize(self._compute_delta_h)(curr_layer.output, 1 - curr_layer.output), weighted_sum
            )

            curr_layer.error_signal = layer_err

            # Construct weight update matrix
            weight_change = learning_rate * np.tile(next_layer.output, (layer_err.size, 1)) * layer_err[:, np.newaxis] + (curr_layer.weight_change * momentum)

            curr_layer.weight_change += weight_change
            # self.layers[index].weights = curr_layer.weights + weight_change

    def train(
        self,
        training_data: pd.DataFrame,
        learning_rate: float,
        momentum: float,
        epochs: int,
        batch_size: int
    ):
        # Split training data into batches
        num_batches = len(training_data.index) / batch_size
        batched_data = np.array_split(training_data.sample(frac=1), num_batches)

        for epoch in range(epochs):
            # Run backpropagation with batches
            for batch in batched_data:
                # Run backpropagation on the batch
                for sample in batch.iterrows():
                    input = sample[1].drop("class").to_numpy()
                    target = [0] * 4
                    target[sample[1].loc["class"] - 1] = 1

                    output = self._feed_forward(input)
                    output_err = np.vectorize(self._compute_delta)(output, target)
                    
                    self.layers[-1].error_signal = output_err

                    if(len(self.layers) > 2):
                        self._back_propagate(learning_rate, momentum)

                    weight_change = learning_rate * np.outer(
                        output_err, self.layers[-2].output
                    )

                    self.layers[-1].weight_change += weight_change
                    # self.layers[-1].weights = self.layers[-1].weights + weight_change

                # Apply the weights from the batch
                map(lambda x: x.update_weights, self.layers)

            print(f"Finished Epoch: {epoch}")


        test = [0,1,2,1,0,3,1,1,0,2,1,1,0,2,2,0,0,0,1,0,1,2,0,0,0,0,0,3,4,0,0,0,0,0,1]
        self._feed_forward(test)
        print(self.layers[-1].output)

    def predict():
        pass

def Sigmoid(action_potential: float) -> float:
    return 1 / (1 + np.exp(-action_potential))


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
    hidden_layer1 = Layer(2, 35, 0.0, np.vectorize(Sigmoid))
    output_layer = Layer(4, 2, 0.0, np.vectorize(Sigmoid))

    NN = NeuralNetwork([input_layer, hidden_layer1, output_layer])

    NN.train(PP.data, 0.01, 0.1, 100, 10)
