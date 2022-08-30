from typing import List, Callable
import pandas as pd

Converter = dict[str, Callable]
Bins = dict[str, int]

class Preprocessor():
    data: pd.DataFrame = None

    def load_raw_data_from_file(self, file_path: str, column_names: List[str], converters: Converter = dict(), bins: Bins = dict()):
        """Load the raw data from a CSV file

        Parameters
        ----------
        file_path: str
            A CSV file containing the unprocessed dataset.

        column_names: List[str]
            A list of column header names

        (Optional) converters: dict[str: Callable]
            A dictionary where the keys are column names
            and the values are functions that modify the
            column values

       (Optional) bins: dict[str: int]
            A dictionary where the keys are column names
            and the values are the number of bins to place
            that column's values into equally spaced.
        """

        # Read the CSV file
        self.data = pd.read_csv(file_path, names=column_names, converters=converters)
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
        self.data.to_csv(path_or_buf=file_path,  header=self.data.columns, index=False)

    def load_processed_data_from_file(self, file_path: str):
        """Load the processed data from a CSV file

        Parameters
        ----------
        file_path: str
            A file path to read the processed data from
        """

        # Read the data in from a CSV file
        self.data = pd.read_csv(file_path, header=0)
        