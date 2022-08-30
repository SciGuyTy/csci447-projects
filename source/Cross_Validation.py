from typing import Callable
import pandas as pd
import numpy as np

import Algorithm

# One possible test is to stratify the class distribution


class Cross_Validation:
    data: pd.DataFrame = None

    def validate(
        self,
        data: pd.DataFrame,
        classification_label: str,
        model: Callable,
        loss_function: Callable,
        num_folds: int = 10,
    ):
        """Perform cross-validation on k folds

        Parameters
        ----------
        data: pd.DataFrame
        The data to perform cross-validation on.

        algorithm: Callable
        The algorithm to evaluate.

        (Optional) num_folds: int
        The number of times to sample from the dataset (defaults to 10).
        """

        # Shuffle the data
        self.data = data.sample(frac=1)

        # Divide the data into k folds
        folded_data = np.array_split(self.data, num_folds)

        total_loss = None

        # Iterate through each fold and run the model
        for index, fold in enumerate(folded_data):
            # Array to store prediction results for this fold
            prediction_results = []

            # Define the data for testing
            test_data = fold

            # Define the data for training
            training_data = folded_data.copy()
            training_data.pop(index)

            # Concatenate training data
            training_data = pd.concat(training_data)

            # Perform prediction using the model
            for _, sample in test_data.iterrows():
                prediction = model(training_data, sample)
                prediction_results.append([sample[classification_label], prediction])

            # Implement the loss function
            total_loss += loss_function(prediction_results)

        # Return the average loss value
        return total_loss / num_folds


df = pd.DataFrame([[1, 2, 3, 4], [5, 6, 7, 8]], columns=["a", "b", "c", "d"])

test = Cross_Validation()


def model(training_data, sample):
    algorithm = Algorithm(training_data, "class")
    return algorithm.predict()


def loss():
    pass


test.validate(df, "class", model, loss, 2)
