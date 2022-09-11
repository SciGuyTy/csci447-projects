from typing import Callable, Union, Any
import pandas as pd
import numpy as np

Column_label = Union[str, int]


class CrossValidation:
    def __init__(
        self,
        data: pd.DataFrame,
        classification_label: Column_label = "class",
        positive_class_value: Any = True,
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
        """

        # Store the data to be used for training and evaluating the model(s)
        self.data: pd.DataFrame = data

        # Store the classification column name
        self.classification_column_name: Column_label = classification_label

        # Store the classification positive value
        self.positive_class_value = positive_class_value

    def __fold_data(self, num_folds: int, stratify: bool):
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
        # Define the levels of classification in the data
        classification_levels = self.data[self.classification_column_name].unique()

        # Shuffle the data into a random order
        shuffled_data = self.data.sample(frac=1)

        # A list to hold the folded data
        folded_data = [None] * num_folds
        
        if(stratify):
            # If the data is to be stratified, iterate through each classification level and divide the data into equally sized chunks based on the number of folds
            for classification in classification_levels:
                class_data = shuffled_data[shuffled_data[self.classification_column_name] == classification]

                # Divide the data for a given class into equally sized chunks
                split_data = np.array_split(class_data, num_folds)

                # "Stack" each kth chunk of data with its counterparts from the other classes
                for index, data in enumerate(split_data):
                    folded_data[index] = pd.concat([data, folded_data[index]])

            # Return the folded data
            return folded_data

        else:
            # If the data is not to be stratified, simply return a list of equally sized chunks of data from the dataset
            return np.array_split(shuffled_data, num_folds)

    def validate(
        self, 
        model: Callable,
        num_folds: int = 10,
        stratify: bool = False,
        alter_data: bool = False
    ) -> float:
        """Perform cross-validation using k=num_folds folds

        Parameters
        ----------
        model: Callable
            The model to perform cross-validation on.

        (Optional) num_folds: int
            The number of times to sample from the dataset (defaults to 10).

        (Optional) alter_data: bool
            Whether to stratify the training data (defaults to False).

        (Optional) alter_data: bool
            Whether to alter the training data (defaults to False).
        """

        proportion_to_shuffle = 0.1

        # Divide the data into k folds
        folded_data = self.__fold_data(num_folds, stratify)

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

            if alter_data:
                # Alter 10% of the training dataset
                self.alter_dataset(training_data, proportion_to_shuffle)

            algorithm = model(training_data, self.classification_column_name)
            algorithm.train()

            # Perform prediction on all samples for this test fold
            for _, sample in test_data.iterrows():
                # Train and execute the model on the given training data and testing data
                prediction = algorithm.predict(sample)

                # Determine whether the prediction is a true positive, false positive, true negative, or false negative
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

    def alter_dataset(self, data: pd.DataFrame, proportion_to_alter: float):
        """Alters the data set by shuffling column values
            in a sample based on the proportion

        Parameters
        ----------
        data: pd.DataFrame
            The data to alter

        proportion_to_alter: float
            The proportion of the data to alter [0, 1]
        """
        # Get a random sample
        altered_data = data.sample(frac=proportion_to_alter)

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
        data.update(altered_data)