from typing import Callable, Union
import pandas as pd
import numpy as np

Column_label = Union[str, int]


class Cross_validation:
    data: pd.DataFrame = None
    classification_column: pd.Series = None

    def set_data(
        self, data: pd.DataFrame, classification_label: Column_label = "class"
    ) -> None:
        """Set the dataset to be used for cross-validating a model

        Parameters
        ----------
        data: pd.DataFrame
            The dataset to be used for cross-validating a model.

        (Optional) classification_label: Column_label
            The label (or index) representing the column storing classification data (defaults to "class").
        """

        # Store the data to be used for training and evaluating the model(s)
        self.data = data

        # Store the column which stores classification data
        self.classification_column = data[classification_label]

    def validate(
        self,
        model: Callable,
        loss_function: Callable,
        num_folds: int = 10,
    ) -> float:
        """Perform cross-validation using k=num_folds folds

        Parameters
        ----------
        model: Callable
            The model to perform cross-validation on.

        loss_function: Callable
            The loss function to use.

        (Optional) num_folds: int
            The number of times to sample from the dataset (defaults to 10).
        """

        # Shuffle the data
        self.data = self.data.sample(frac=1)

        # Divide the data into k folds
        folded_data = np.array_split(self.data, num_folds)

        # Total loss value (used to calculate average loss value)
        total_loss = 0

        # Iterate through each fold and run the model
        for index, fold in enumerate(folded_data):
            # Array to store prediction results for this fold
            prediction_results = pd.DataFrame(columns=["actual", "prediction"])

            # Define the data for testing (a single fold)
            test_data = fold

            # Define the data for training (remaining folds)
            training_data = folded_data.copy()
            training_data.pop(index)

            # Combine training data into a single DataFrame
            training_data = pd.concat(training_data)

            # Perform prediction on all samples for this test fold
            for _, sample in test_data.iterrows():
                # Train and execute the model on the given training data and testing data
                prediction_results.loc[len(prediction_results)] = model(
                    training_data, sample
                )

            # Implement the loss function and update the total loss value
            total_loss += loss_function(prediction_results)

        # Return the average loss value
        return total_loss / num_folds
