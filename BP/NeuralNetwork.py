from Preprocess import Preprocessor
from Layer import Layer
import numpy as np

class NeuralNetwork():
    def __init__(self, layers) -> None:
        self.layers = layers

    def feed_forward(self, input):
        prev_output = input

        for layer in self.layers[1:len(self.layers)]:
            layer.compute_output(prev_output)
            prev_output = layer.output

    def backpropagate(self, target, learning_rate, momentum):
        self.layers[-1].delta = 2 * np.subtract(self.layers[-1].output, target)

        for index in range(len(self.layers) - 2, 0, -1):
            # Get reference to current, preceding, and following layer
            curr_layer = self.layers[index]
            prev_layer = self.layers[index + 1]

            curr_layer.delta = np.matmul(prev_layer.delta, prev_layer.weights.T)

            curr_layer.weights -= learning_rate * curr_layer.delta
            # curr_layer.weight_change.append((learning_rate * curr_layer.delta * curr_layer.output))

    def train(self, training_data, target_feature, epochs, batch_size, learning_rate, momentum):
        for i in range(epochs):
            classes = training_data[target_feature].unique()

            features = training_data.columns.drop(target_feature)
            
            # num_batches = len(training_data.index) / batch_size
            # batched_data = np.array_split(training_data.sample(frac=1), num_batches)

            # for batch in batched_data:
            for sample in training_data.iterrows():
                _, pattern = sample

                target = [0] * len(classes)
                target[pattern.loc[target_feature] - 1] = 1

                self.feed_forward(pattern[features])
                self.backpropagate(target, learning_rate, momentum)
            
                # Apply the weight change for each layer
                # for layer in self.layers:
                    # layer.update_weights()

if __name__ == "__main__":
    def sigmoid(x):
        return (1 / (1 + np.exp(-x)))

    def softmax(x):
        softmax_vector = np.exp(x) / np.exp(x).sum()
        bounded = ([0] * len(x))
        bounded[np.argmax(softmax_vector)] = 1
        return bounded

    file_path = "../datasets/classification/test-classification.data"

    # Process the data
    PP = Preprocessor()
    PP.load_raw_data_from_file(file_path, ["a", "b", "class"])


    layer1 = Layer(2)
    layer2 = Layer(5, layer1, activation=np.vectorize(sigmoid))
    layer3 = Layer(5, layer2, activation=np.vectorize(sigmoid))
    layer4 = Layer(2, layer3, activation=softmax)

    layers = [layer1, layer2, layer3, layer4]

    network = NeuralNetwork(layers)
    network.train(PP.data, "class", 50, 1, 0.1, 0.2)

    network.feed_forward([1, 0])
    res = layer4.output

    print(layer3.output)
    print(layer4.output)

