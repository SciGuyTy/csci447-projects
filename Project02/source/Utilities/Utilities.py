import math
from typing import List

import pandas as pd


class Utilities:

    @classmethod
    def get_normalization_parameters(cls, data: pd.DataFrame, target_feature: str = "class"):
        training_features = data.columns.drop(target_feature)
        normalization_parameters = pd.DataFrame(columns=["feature", "min", "max"])
        for feature in training_features:
            data[feature] = data[feature].astype("float64")
        pass



    @classmethod
    def normalize(cls, data: pd.DataFrame, target_feature: str = "class"):
        """
        Perform min-max normalization on a set of features belonging to the dataset
        (i.e., bound the feature values between 0 and 1, inclusive). Return the
        normalization parameters for the features.

        Parameters
        -----------
        data: pd.DataFrame
            The data with which to normalize

        target_feature: str
            The response feature of the data set (defaults to "class")
        """
        # Retrieve training features from data
        training_features = data.columns.drop(target_feature)

        # Instantiate normalization parameters
        normalization_parameters = pd.DataFrame(columns=["feature", "min", "max"])

        for feature in training_features:

            # If the feature is one hot encoded
            if data[feature].dtype.kind == 'O':
                continue

            # Cast dtype of features to be normalized as a float
            data[feature] = data[feature].astype("float64")

            # Retrieve the minimum and maximum values for the given column
            minimum_value = data[feature].min()
            maximum_value = data[feature].max()

            # Add the normalization parameters to the data frame.
            normalization_parameters.loc[len(normalization_parameters.index)] = [feature, minimum_value, maximum_value]

            # Define the newly normalized feature in the dataset, and assign its values to
            # the normalized version of the original data

            data[feature] = (data[feature] - minimum_value) / (
                maximum_value - minimum_value
            )

        # Return the normalization parameters
        return normalization_parameters

    @classmethod
    def normalize_individual_by_params(cls, instance, norm_params: pd.DataFrame):
        """
        Perform min-max normalization on a single instance belonging to the dataset

        Parameters
        -----------
        instance
            The data with which to normalize

        """

        # Loop through each feature to normalize
        for index, row in norm_params.iterrows():
            feature = row["feature"]

            # If the feature is one hot encoded
            if instance[feature].dtype == object:
                continue

            # Normalize the instance's feature
            instance[feature] = (instance[feature] - row['min']) / (
                row['max'] - row['min']
            )

        # Return the normalized instance
        return instance

    @classmethod
    def normalize_set_by_params(cls, data: pd.DataFrame, norm_params: pd.DataFrame):
        """
        Perform min-max normalization on a group of instances belonging to the dataset

        Parameters
        -----------
        data
            The data with which to normalize

        """
        for index, row in norm_params.iterrows():
            feature = row["feature"]

            # If the feature is one hot encoded
            if data[feature].dtype.kind == 'O':
                continue

            # Cast dtype of features to be normalized as a float
            data[feature] = data[feature].astype("float64")

            # Retrieve the minimum and maximum values for the given column
            minimum_value = row['min']
            maximum_value = row['max']

            # Define the newly normalized feature in the dataset, and assign its values to
            # the normalized version of the original data

            if maximum_value - minimum_value == 0:
                data[feature] = 0
            else:
                data[feature] = (data[feature] - minimum_value) / (
                    maximum_value - minimum_value
                )
            

        # Return the normalized data
        return data





class OneHotEncoding:

    root_2 = math.sqrt(2)
    def __init__(self, value):
        self.value = value

    def __sub__(self, other):
        if self.value == other.value:
            return 0
        return self.root_2

class CyclicalData:

    def __init__(self, value, total):
        self.value = value
        self.total = total

    def __sub__(self, other):
        return min((self.value - other.value) % 12, (other.value-self.value) % 12)