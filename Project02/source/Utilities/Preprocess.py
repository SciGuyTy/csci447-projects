import math
from typing import List, Callable
import pandas as pd

Converter = dict[str, Callable]
Bins = dict[str, int]


class Preprocessor:
    data: pd.DataFrame = None

    @staticmethod
    def float_converter(na_values):
        def value_to_float(x):
            if x in na_values:
                return x
            return float(x)

        return value_to_float

    def load_raw_data_from_file(
        self,
        file_path: str,
        column_names: List[str],
        columns_to_drop: List[str] = [],
        converters: Converter = dict(),
        bins: Bins = dict(),
        dropNA: List[str] = False,
        columns_to_floats = []
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


        [converters.update({col: self.float_converter(dropNA)}) for col in columns_to_floats]

        # Read the CSV file
        self.data = pd.read_csv(file_path, names=column_names, converters=converters)

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
