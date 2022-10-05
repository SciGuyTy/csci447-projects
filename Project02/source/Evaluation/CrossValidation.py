from Project02.source.Utilities.Utilities import Utilities
from typing import Callable, Union

import pandas as pd
import numpy as np


Column_label = Union[str, int]


class CrossValidation:
    def __init__(
        self,
        data: pd.DataFrame,
        target_feature: Column_label = "class",
        regression: bool = False,
    ):
        """Set the dataset to be used for cross-validating a model

        Parameters
        ----------
        data: pd.DataFrame
            The dataset to be used for cross-validating a model.

        (Optional) classification_label: Column_label
            The label (or index) representing the column storing classification data (defaults to "class").

        (Optional) positive_class_value: Any
            The value representing a positive outcome for the classification column (defaults to True).

        (Optional) target_feature: Column_label
            The label of the feature which is being used as the target (defaults to "class"),

        (Optional) regression: bool
            Whether the underlying data is being used for regression (defaults to False),
        """

        # Store the data to be used for training and evaluating the model(s)
        self.data: pd.DataFrame = data

        # Store the classification column name
        self.target_feature = target_feature

        # The algorithm
        self.algorithm = None

        # Whether the underlying data is being used for regression
        self.regression = regression

        if not self.regression:
            # Get a list of all class levels
            self.classes = self.data[self.target_feature].unique()

    def fold_data(self, num_folds: int, stratify: bool):
        """Divide a dataset into folds (for cross-validation)

        Parameters
        ----------
        num_folds: int
            The number of folds (number of 'chunks' with which to split the data).

        stratify: bool
            Whether the given folds should be stratified (i.e., split in a manner such that
            the proportion of classifications within the dataset is similarly represented in each fold).

        Returns
        -------
        folded_data: List[pd.DataFrame]
            A list of k=num_folds DataFrames containing a fold of data

        """
        # Shuffle the data into a random order
        shuffled_data = self.data.sample(frac=1)

        # A list to hold the folded data
        folded_data = [pd.DataFrame(columns=self.data.columns)] * num_folds

        if not stratify:
            # If the data is not to be stratified, simply return a list of equally sized
            # chunks of data from the dataset
            return np.array_split(shuffled_data, num_folds)

        if self.regression:
            # Sort the data based on the target feature
            sorted_data = shuffled_data.sort_values(self.target_feature)

            fold_size = int(len(sorted_data.index) / num_folds)

            # Decompose data into 10 'bins' of consecutive data
            split_data = np.array_split(sorted_data, fold_size)

            for fold_id in range(len(folded_data)):
                for group in split_data:
                    # For each fold, select the corresponding value from each group in the
                    # split value array (so for example, the first fold will grab the first
                    # element from each group in the split value array)
                    folded_data[fold_id] = pd.concat(
                        [folded_data[fold_id], group.iloc[fold_id].to_frame().T],
                        ignore_index=True,
                    )

        else:
            # Define the levels of classification in the data
            classification_levels = self.classes

            # If the data is to be stratified, iterate through each classification level
            # and divide the data into equally sized chunks based on the number of folds
            for classification in classification_levels:
                class_data = shuffled_data[
                    shuffled_data[self.target_feature] == classification
                ]

                # Divide the data for a given class into equally sized chunks
                split_data = np.array_split(class_data, num_folds)

                # "Stack" each kth chunk of data with its counterparts from the other classes
                for index, data in enumerate(split_data):
                    folded_data[index] = pd.concat([data, folded_data[index]])

        # Return the folded data
        return folded_data

    def validate(
        self,
        model: Callable,
        num_folds: int = 10,
        stratify: bool = False,
        model_params=[],
        predict_params=[],
    ) -> float:
        """Perform cross-validation using k=num_folds folds

        Parameters
        ----------
        model: Callable
            The model to perform cross-validation on.

        (Optional) num_folds: int
            The number of times to sample from the dataset (defaults to 10).

        (Optional) alter_data: bool
            Whether to alter the training data (defaults to False).
        """

        # Divide the data into k folds
        folded_data = self.fold_data(num_folds, stratify)

        # Get the training and test data pairs for each fold
        training_test_data = self.get_training_test_data_from_folds(folded_data)

        # Results for all the folds
        overall_results = []

        # Iterate through the training and test data pairs
        for training_data, test_data, _ in training_test_data:
            # Instantiate the model
            self.algorithm = model(training_data, self.target_feature, *model_params)

            fold_results = self.calculate_results_for_fold(self.algorithm, test_data, predict_params=predict_params)

            overall_results.append(fold_results)

        # Return the average loss value
        return overall_results

    def get_training_test_data_from_folds(self, folded_data):
        training_test_data = []

        # Iterate through each fold
        for index, fold in enumerate(folded_data):
            # Define the data for testing (a single fold)
            test_data = fold.copy()

            # Define the data for training (remaining folds)
            training_data = folded_data.copy()
            training_data.pop(index)

            # Combine training data into a single DataFrame
            training_data = pd.concat(training_data)

            # Normalize the training and testing data
            norm_params = Utilities.normalize(training_data, self.target_feature)
            test_data = Utilities.normalize_set_by_params(test_data, norm_params)

            # Add the training and test data as a pair to the list
            training_test_data.append((training_data, test_data, norm_params))

        return training_test_data

    def calculate_results_for_fold(self, algorithm, test_data, predict_params=[]):
        if self.regression:
            # DataFrame to store results for this fold
            fold_results = pd.DataFrame(columns=["actual", "predicted"])

            for sample_index, sample in test_data.iterrows():
                # Train and execute the model on the given training data and testing data
                prediction = algorithm.predict(sample, *predict_params)

                # Append the prediction and actual values to the fold_results
                fold_results.loc[sample_index] = [
                    sample[self.target_feature],
                    prediction,
                ]
        else:
            # DataFrame to store results for this fold
            fold_results = pd.DataFrame(0, columns=self.classes, index=self.classes)

            # Perform prediction on all samples for this test fold
            for sample_index, sample in test_data.iterrows():
                # Train and execute the model on the given training data and testing data
                actual = sample[self.target_feature]
                prediction = algorithm.predict(sample, *predict_params)
                # Increment the prediction/actual pair in the confusion matrix
                fold_results[actual][prediction] += 1

        return fold_results