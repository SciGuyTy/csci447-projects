from typing import List

import pandas as pd


class Utilities:
    @classmethod
    def normalize(self, data: pd.DataFrame, target_feature: str = "class"):
        """
        Perform min-max normalization on a set of features belonging to the dataset
        (i.e., bound the feature values between 0 and 1, inclusive)

        Parameters
        -----------
        data: pd.DataFrame
            The data with which to normalize

        target_feature: str
            The response feature of the data set (defaults to "class")
        """
        # Retrieve training features from data
        training_features = data.columns.drop(target_feature)

        for feature in training_features:
            # Cast dtype of features to be normalized as a float
            data[feature] = data[feature].astype("float64")

            # Retrieve the minimum and maximum values for the given column
            minimum_value = data[feature].min()
            maximum_value = data[feature].max()

            # Define the newly normalized feature in the dataset, and assign its values to
            # the normalized version of the original data

            data[feature] = (data[feature] - minimum_value) / (
                maximum_value - minimum_value
            )

        # Return the normalized data
        return data
