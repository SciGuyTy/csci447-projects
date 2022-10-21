from Project03.Algorithm.NeuralNetwork import NeuralNetwork as NN
import numpy as np
def test_nn():
    nn = NN([5, 5, 5, 2])

    x = np.array([1, 2, 3, 4, 5])

    print(nn.calculate_feed_forward(x))


if __name__ == "__main__":
    test_nn()