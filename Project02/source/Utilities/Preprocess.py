from cmath import pi
import math
from typing import Dict, List, Callable
import pandas as pd
import numpy as np

Converter = dict[str, Callable]
Bins = dict[str, int]


class Preprocessor:
    data: pd.DataFrame = None


    def load_raw_data_from_file(
        self,
        file_path: str,
        column_names: List[str],
        columns_to_drop: List[str] = [],
        converters: Converter = dict(),
        bins: Bins = dict(),
        dropNA: List[str] = False,
        cyclical_features: Dict[str, int] = {}
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

        (Optional) cyclical_features: Dict[str, int]
            A list of features that contain cyclical data and should be 
            encoded using sine/cosine transformations. Defaults to an empty
            dictionary. Note, the keys in the dictionary represent feature 
            names, and the values represent the period of the data.
        """

        # Read the CSV file
        self.data = pd.read_csv(file_path, names=column_names, converters=converters)

        # Encode cyclical features using sine/cosine transformations
        for feature, period in cyclical_features.items():
            # Reference to the unencoded feature values
            feature_values = self.data[feature]

            # Index of feature column in data (used when inserting transformed features)
            feature_loc = self.data.columns.get_loc(feature)

            # Map the values using sine and cosine functions
            sin_transform = np.sin((2 * np.pi * feature_values) / period)
            cos_transform = np.cos((2 * np.pi * feature_values) / period)

            # Insert the two new features
            self.data.insert(feature_loc, f"sin_{feature}", sin_transform)
            self.data.insert(feature_loc + 1, f"cos_{feature}", cos_transform)

        # Drop unwanted columns
        self.data = self.data.drop(columns=columns_to_drop)

        # Drop rows containing missing values
        if dropNA:
            self.data = self.data.replace(dict.fromkeys(dropNA, math.nan))
            self.data = self.data.dropna()

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
