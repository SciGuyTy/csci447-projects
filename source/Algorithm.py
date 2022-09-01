from typing import Any, Dict, Union
import pandas as pd

# Type definitions
Column_label = Union[int, str]
Observation = Dict[Column_label, int]
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
            A Pandas DataFrame object containing the training dataset

        classification_label: Column_label
            The label (or index) representing the column storing classification data
        """

        # Reference to the column containing classification data for each sample
        self.classification_column = training_data[classification_label]

        # Reference to the actual training data
        self.training_data = training_data

        # A list of possible classifications within the data
        self.training_classes = self.classification_column.unique()

        # A list of attributes representing the objects of interest
        self.training_attributes = training_data.columns.drop(classification_label)

        # A dictionary which contains the marginal probabilities for every classification level
        self.classification_probability = {}

        # A dictionary which contains the conditional probabilities for a given classification-attribute pair
        self.training_distribution = {}

    def __compute_classification_probability(self, classification: Any) -> float:
        """Compute the marginal probability for an observation to exist within a specified
        classification level (i.e., the prior)

        Parameters
        ----------
        classification: Classification
            The classification with which to compute the relative proportion of observations

        Returns
        -------
        proportion: float
            The proportion of observations with the provided classification
            relative to the total number of observations in the dataset
        """
        # Compute the number of total samples
        num_total_samples = len(self.training_data)

        # Compute the number of samples in the classification sub-set
        num_samples_in_class = len(
            self.training_data[self.classification_column == classification]
        )

        # Return the marginal probability for a sample to be classified within the given classification level
        return num_samples_in_class / num_total_samples

    def __compute_attribute_probability(
        self,
        classification: Any,
        attribute_label: Column_label,
        attribute_value: int,
    ) -> float:
        """Compute the conditional probability for an attribute having a given value
        for a given classification level

        Parameters
        ----------
        classification: Classification
            The classification with which the observations belong to

        attribute_label: Union[int, str]
            The attribute with which to compute the proportion of observations whose
            values are equal to that of the novel observation

        attribute_value: int
            The observed (new) value for the given attribute

        Returns
        -------
        proportion: float
            A modified proportion of samples whose value for the given attribute matches the given 
            attribute value relative to the total number of observations in the given classification level
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

        # Compute the number of training attributes in these data
        num_attributes = len(self.training_attributes)

        # Return the conditional probability of 
        return (num_equal_attributes + 1) / (num_samples_in_class + num_attributes)

    def train(self):
        """Train the model on a set of data by computing the probability of
        some data attribute given some classification (for all combinations
        of data attribute and classification)
        """

        # Loop through each classification level and compute the marginal probability
        for classification in self.training_classes:
            self.classification_probability[
                classification
            ] = self.__compute_classification_probability(classification)

            # For each classification level, loop through each training attribute and get a list of all unique recorded values for the given attribute
            for attribute in self.training_attributes:
                attribute_values = self.training_data[attribute].unique()

                # For each unique value of the given attribute...
                for value in attribute_values:
                    # Compute the conditional probability of observing this classification given this combination of attribute and attribute value
                    likelihood = self.__compute_attribute_probability(
                        classification, attribute, value
                    )

                    # Record the conditional probability into a dictionary for later reference when making predictions
                    self.training_distribution[
                        (classification, attribute, value)
                    ] = likelihood

    def predict(self, observation: Observation) -> Prediction:
        """Classify a novel observation based on the observation's attributes

        observation: Observation
            A novel observation with which to classify based on attribute values.]
            The attribute values should be stored as a dictionary with keys that match
            column names for the dataset

        Returns
        -------
        prediction: Classification
            The predicted classification for the provided observation
        """

        # Scoring object that contains prediction results
        pred_results = self.classification_probability.copy()

        for classification in pred_results:
            for attribute in observation:
                # Multiply the current conditional probability by the existing conditional probability
                pred_results[classification] *= self.training_distribution[
                    (classification, attribute, observation[attribute])
                ]

        # Return the classification label whose conditional probability given the observation's attributes is the largest
        return max(pred_results, key=pred_results.get)
