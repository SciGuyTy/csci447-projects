from typing import List, Callable
import pandas as pd

Converter = dict[str, Callable]
Bins = dict[str, int]

class Preprocessor():
    data: pd.DataFrame = None

    def load_raw_data_from_file(self, file_path: str, column_names: List[str], columns_to_drop=[], converters: Converter = dict(), bins: Bins = dict()):
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

        # Drop unwanted cols
        self.data = self.data.drop(columns=columns_to_drop)

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

    def alter_dataset(self, proportion_to_alter: float):
        """Alters the data set by shuffling column values
            in a sample based on the proportion

        Parameters
        ----------
        proportion_to_alter: float
            The proportion of the data to alter [0, 1]
        """
        # Get a random sample
        altered_data = self.data.sample(frac=proportion_to_alter)

        # Loop through each column to shuffle it
        for col_name in altered_data.columns:
            # Get a new version of the column as a data frame with two columns (index, col_name)
            # so we have a new set of temporary indices on the rows
            shuffled_col_df = altered_data[col_name].reset_index()

            # Shuffle the col_name column of the new df and reset the index colum, creating just a series
            shuffled_col = shuffled_col_df[col_name].sample(frac=1).reset_index(drop=True)

            # Assign the shuffled column series to the created df
            shuffled_col_df[col_name] = shuffled_col

            # Remove the temporary indices and restore the original
            shuffled_col_df = shuffled_col_df.set_index('index', drop=True)

            # Update the sample to have the shuffled column
            altered_data[col_name] = shuffled_col_df

        # Update the data with the shuffled sample
        self.data.update(altered_data)


