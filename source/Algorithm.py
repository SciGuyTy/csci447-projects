from typing import Any, Dict, List, Tuple, Union
from unicodedata import name
import pandas as pd

Column_label = Union[int, str]

Classification = Any
Attribute = List[Column_label]

Observation = Dict[Attribute, int]
Prediction = Dict[Column_label, float]


class Algorithm:
    def __init__(
        self, training_data: pd.DataFrame, classification_label: Column_label
    ) -> None:
        """Initialize the algorithm by decomposing training classifications and and attributes
        from the training data

        Parameters
        ----------
        training_data: pandas.DataFrame
            A Pandas DataFrame object containing the training dataset.

        classification_label: Column_label
            The label (or index) representing the column storing classification data.
        """

        # Reference to the column containing classification data for each sample
        self.classification_column = training_data[classification_label]

        # Reference to the actual training data
        self.training_data = training_data

        # An array of possible classifications within the data
        self.training_classes = self.classification_column.unique()

        # An array of attributes representing the objects of interest
        self.training_attributes = training_data.columns.drop(classification_label)

    def __get_classification_proportion(self, classification: Classification) -> float:
        """Compute the proportion of observations with the provided classification
        relative to the total number of observations in the dataset

        Parameters
        ----------
        classification: Classification
            The classification with which to compute the relative proportion of observations.

        Returns
        -------
        proportion: float
            The proportion of observations with the provided classification
            relative to the total number of observations in the dataset.
        """
        # Compute the number of total samples
        num_total_samples = len(self.training_data)

        # Compute the number of samples in the classification sub-set
        num_samples_in_class = len(
            self.training_data[self.classification_column == classification]
        )

        return num_samples_in_class / num_total_samples

    def __get_attribute_score(
        self,
        classification: Classification,
        attribute_label: Union[int, str],
        attribute_value: int,
    ) -> float:
        """Compute the proportion of observations within a classification
        that have a value equal to the novel observation for a given attribute

        Parameters
        ----------
        classification: Classification
            The classification with which the observations belong to.

        attribute_label: Union[int, str]
            The attribute with which to compute the proportion of observations whose
            values are equal to that of the novel observation.

        attribute_value: int
            The observed (new) value for the given attribute

        Returns
        -------
        proportion: float
            The proportion of observations within a classification that have a
            value equal to the novel observation for a given attribute.
        """
        # Compute the number of samples in the classification sub-set
        num_samples_in_class = len(
            self.training_data[self.classification_column == classification]
        )

        # Expression used to find samples that belong to the given glass and have the same value as the observation for a given attribute
        condition = (self.training_data["class"] == classification) & (
            self.training_data[attribute_label] == attribute_value
        )

        # Compute the number of samples in the classification sub-set that match the condition
        num_equal_attributes = len(self.training_data[condition])

        # Compute the number of attributes
        num_attributes = len(self.training_attributes)

        return (num_equal_attributes + 1) / (num_samples_in_class + num_attributes)

    def predict(self, observation: Observation) -> Prediction:
        """Classify a novel observation based on the observation's attributes

        observation: Observation
            A novel observation with which to classify based on attribute values.]
            The attribute values should be stored as a dictionary with keys that match
            column names for the dataset.

        Returns
        -------
        prediction: Classification
            The predicted classification for the provided observation.
        """

        # Scoring object that contains prediction results
        pred_results = dict.fromkeys(self.training_classes)

        for classification in self.training_classes:
            # Set the score for each classification to the proportion of samples within said classification
            pred_results[classification] = self.__get_classification_proportion(
                classification
            )

            for attribute in self.training_attributes:
                # Multiply the current 'classification score' by the 'attribute score'
                pred_results[classification] *= self.__get_attribute_score(
                    classification, attribute, observation[attribute]
                )

        return pred_results
