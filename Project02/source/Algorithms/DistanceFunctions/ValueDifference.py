from numbers import Number
from source.Algorithms.DistanceFunctions.DistanceFunction import DistanceFunction

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd


class ValueDifference(DistanceFunction):
    def __init__(self, data: pd.DataFrame, class_feature: str, p: int = 2) -> None:
        """
        The Value Difference Metric for discrete valued features

        Parameters:
        -----------
        data: pd.DataFrame
            The data set with which the vectors used for VDM calculation belong

        class_column: str
            The name of the feature which is used to identify classification level

        p: int
            The parameter used to influence the VDM
        """
        self.p = p
        self.data = data
        self.class_feature = class_feature

    def compute_distance(self, v1: pd.Series, v2: pd.Series) -> float:
        """
        Compute the Value Difference Metric for two vectors with discrete features

        Parameters:
        -----------
        v1: pd.Series
            The first vector in the computation

        v1: pd.Series
            The second vector in the computation

        Returns:
        --------
        A float representing the VDM for v1 and v2
        """

        # Run default validation/preprocessing defined by the DistanceFunction interface
        super().compute_distance(v1, v2)

        dist = 0

        # Retrieve classification levels from data
        classification_levels = self.data[self.class_feature].unique()

        # Compute the value difference metric
        for attribute in v1.index:
            attr_dist = 0

            # For each attribute, compute the difference in class conditional probability
            # for the given attribute values of v1, v2 on each class
            for classification_level in classification_levels:
                # Class conditional probability for the given attribute on v1 P(c|v1_a)
                v1_class_conditional = self._compute_class_conditional(
                    classification_level, attribute, v1[attribute]
                )

                # Class conditional probability for the given attribute on v1 P(c|v2_a)
                v2_class_conditional = self._compute_class_conditional(
                    classification_level, attribute, v2[attribute]
                )

                # Difference between the two probabilities of order p
                attr_dist += abs(v1_class_conditional - v2_class_conditional) ** self.p

            dist += attr_dist

        return dist

    def _compute_class_conditional(
        self, classification: str, attribute_label: str, attribute_value: Number
    ) -> float:
        """
        Compute the class conditional probability for a given attribute and classification level

        Parameters:
        -----------
        classification: str
            The classification level

        attribute_level: str
            The attribute

        attribute_value: Any
            The attribute value to compute CCP for with respect to classification level

        Returns:
        --------
        A float representing the class conditional probability for the given parameters
        """
        # Query condition that describes instances within the data that have the given value
        # for the attribute and belong to the given classification level
        condition = (self.data[self.class_feature] == classification) & (
            self.data[attribute_label] == attribute_value
        )

        # Retrieve the number of instances within the data that match the condition
        num_attr_in_class = len(self.data[condition])

        # Compute the total number of instances within the data that have the given
        # value for the attribute
        num_attr = len(self.data[self.data[attribute_label] == attribute_value])

        # Return the CCP
        return num_attr_in_class / num_attr
