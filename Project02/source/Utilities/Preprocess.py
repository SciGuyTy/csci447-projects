import math
from typing import List, Callable
import pandas as pd

Converter = dict[str, Callable]
Bins = dict[str, int]


class Preprocessor:
    data: pd.DataFrame = None

    def _normalize(self, target_features: List[str]):
        """
        Perform min-max normalization on a set of features belonging to the dataset
        (i.e., bound the feature values between 0 and 1, inclusive)

        Parameters:
        -----------
        target_features: List[str]
            The set of features with which to normalize
        """
        for feature in target_features:
            # Cast dtype of features to be normalized as a float 
            self.data[feature] = self.data[feature].astype("float64")

            # Retrieve the minimum and maximum values for the given column
            minimum_value = self.data[feature].min()
            maximum_value = self.data[feature].max()

            # Define the newly normalized feature in the dataset, and assign its values to
            # the normalized version of the original data

            self.data[feature] = (self.data[feature] - minimum_value) / (
                maximum_value - minimum_value
            )

    def load_raw_data_from_file(
        self,
        file_path: str,
        column_names: List[str],
        columns_to_drop: List[str] = [],
        columns_to_normalize: List[str] = [],
        converters: Converter = dict(),
        bins: Bins = dict(),
        dropNA: List[str] = False,
    ):
        """Load the raw data from a CSV file

        Parameters
        ----------
        file_path: str
            A CSV file containing the unprocessed dataset.

        column_names: List[str]
            A list of column header names

        (Optional) columns_to_drop: List[str]
            A list of columns drop. Defaults to no columns (empty List).

        (Optional) columns_to_normalize: List[str]
            A list of columns to (min-max) normalize. 
            Defaults to no columns (empty List).

        (Optional) converters: dict[str: Callable]
            A dictionary where the keys are column names
            and the values are functions that modify the
            column values

        (Optional) bins: dict[str: int]
            A dictionary where the keys are column names
            and the values are the number of bins to place
            that column's values into equally spaced.
        
        (Optional) dropNA: List[str]
            Whether to drop rows containing missing values. 
            Defaults to False, however providing an array of any length
            will set the flag to true. The contents of the array indicate
            what values should be considered "NA" (in addition to NA values
            defined by python, such as NaN and None).
        """

        # Read the CSV file
        self.data = pd.read_csv(file_path, names=column_names, converters=converters)

        # Drop unwanted columns
        self.data = self.data.drop(columns=columns_to_drop)

        # Drop rows containing missing values
        if dropNA:
            self.data = self.data.replace(dict.fromkeys(dropNA, math.nan))
            self.data = self.data.dropna()

        # Normalize columns
        self._normalize(columns_to_normalize)

        # Bin the data
        for key, value in bins.items():
            self.data[key] = pd.cut(self.data[key], value, labels=False)

    def save_processed_data_to_file(self, file_path: str):
        """Save the processed data to a CSV file

        Parameters
        ----------
        file_path: str
            A file path to save the processed data to
        """

        # Save the data to a CSV file
        self.data.to_csv(path_or_buf=file_path, header=self.data.columns, index=False)

    def load_processed_data_from_file(self, file_path: str):
        """Load the processed data from a CSV file

        Parameters
        ----------
        file_path: str
            A file path to read the processed data from
        """

        # Read the data in from a CSV file
        self.data = pd.read_csv(file_path, header=0)
