import math

from Project03.Utilities.Utilities import Utilities
from typing import Union

import pandas as pd
import numpy as np

Column_label = Union[str, int]


class CrossValidation:
    def __init__(
        self,
        data: pd.DataFrame,
        target_feature: Column_label,
        regression: bool,
    ):
        """Set the dataset to be used for cross-validating a model

        Parameters
        ----------
        data: pd.DataFrame
            The dataset to be used for cross-validating a model.

        target_feature: Column_label
            The label of the feature which is being used as the target

        regression: bool
            Whether the underlying data is being used for regression
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

    def get_tuning_set(self, proportion):
        # Shuffle the data into a random order
        shuffled_data = self.data.sample(frac=1)
        hold_out_size = int(len(shuffled_data.index) * proportion)

        if self.regression:
            # Sort the data based on the target feature
            sorted_data = shuffled_data.sort_values(self.target_feature)


            # Decompose data into 10 'bins' of consecutive data
            split_data = np.array_split(sorted_data, hold_out_size)

            folded_data = [pd.DataFrame(columns=self.data.columns)] * int(1 / proportion)

            fold_id = 0
            for group in split_data:
                # For each fold, select the corresponding value from each group in the
                # split value array (so for example, the first fold will grab the first
                # element from each group in the split value array)
                folded_data[fold_id] = pd.concat(
                    [folded_data[fold_id], group.iloc[fold_id].to_frame().T],
                    ignore_index=True,
                )

            return folded_data[0]

        else:
            # Define the levels of classification in the data
            classification_levels = self.classes

            hold_out = pd.DataFrame(columns=self.data.columns)

            # If the data is to be stratified, iterate through each classification level
            # and divide the data into equally sized chunks based on the number of folds
            for classification in classification_levels:
                class_data = shuffled_data[
                    shuffled_data[self.target_feature] == classification
                ]

                # Divide the data for a given class into equally sized chunks
                split_data = np.array_split(class_data, math.ceil(1/proportion))

                # "Stack" each chunk of data with its counterparts from the other classes
                hold_out = pd.concat([split_data[0], hold_out])

            # Return the folded data
            return hold_out

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

    def validate_for_folds(self, training_test_data, models):
        # Results for all the folds
        overall_results = []

        # Iterate through the training and test data pairs
        for fold, (training_data, test_data, _) in enumerate(training_test_data):
            self.algorithm = models[fold][2]
            fold_results = self.calculate_results_for_fold(self.algorithm, test_data)
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

    def calculate_results_for_fold(self, model, test_data):
        if self.regression:
            # DataFrame to store results for this fold
            fold_results = pd.DataFrame(columns=["actual", "predicted"])

            for sample_index, sample in test_data.iterrows():
                input = sample.drop(self.target_feature).to_numpy()
                prediction = model.predict(input)

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
                input = sample.drop(self.target_feature).to_numpy().astype(float)
                prediction = model.predict(input)
                # Increment the prediction/actual pair in the confusion matrix
                fold_results[actual][prediction] += 1

        return fold_results


