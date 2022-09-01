from typing import Callable, Union
import pandas as pd
import numpy as np

Column_label = Union[str, int]


class CrossValidation:

    def __init__(
        self, data: pd.DataFrame, classification_label: Column_label = "class", positive_class_value=True
    ) :
        """Set the dataset to be used for cross-validating a model

        Parameters
        ----------
        data: pd.DataFrame
            The dataset to be used for cross-validating a model.

        (Optional) classification_label: Column_label
            The label (or index) representing the column storing classification data (defaults to "class").
        """

        # Store the data to be used for training and evaluating the model(s)
        self.data: pd.DataFrame = data

        # Store the column which stores classification data
        self.classification_column: pd.Series = data[classification_label]

        self.classification_column_name: Column_label = classification_label

        self.positive_class_value = positive_class_value

    def validate(
        self,
        model: Callable,
        num_folds: int = 10,
    ) -> float:
        """Perform cross-validation using k=num_folds folds

        Parameters
        ----------
        model: Callable
            The model to perform cross-validation on.

        (Optional) num_folds: int
            The number of times to sample from the dataset (defaults to 10).
        """

        # Shuffle the data
        self.data = self.data.sample(frac=1)

        # Divide the data into k folds
        folded_data = np.array_split(self.data, num_folds)

        # Results for all the folds
        overall_results = []

        # Iterate through each fold and run the model
        for index, fold in enumerate(folded_data):

            # Results for this fold
            fold_results = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}

            # Define the data for testing (a single fold)
            test_data = fold

            # Define the data for training (remaining folds)
            training_data = folded_data.copy()
            training_data.pop(index)

            # Combine training data into a single DataFrame
            training_data = pd.concat(training_data)

            algorithm = model(training_data, self.classification_column_name)
            algorithm.train()

            # Perform prediction on all samples for this test fold
            for _, sample in test_data.iterrows():
                # Train and execute the model on the given training data and testing data
                prediction = algorithm.predict(sample)
                if prediction == self.positive_class_value:
                    if prediction == sample[self.classification_column_name]:
                        fold_results["TP"] += 1
                    else:
                        fold_results["FP"] += 1
                else:
                    if prediction == sample[self.classification_column_name]:
                        fold_results["TN"] += 1
                    else:
                        fold_results["FN"] += 1
            overall_results.append(fold_results)

        # Return the average loss value
        return overall_results
