from source.Algorithms.DistanceFunctions.DistanceFunction import (
    DistanceFunction,
)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd


class Minkowski(DistanceFunction):
    def __init__(self, p: int = 2) -> None:
        """
        The Minkowski metric used to define a number of distance functions

        Parameters:
        -----------
        p: int
            The parameter used to influence the Minkowski metric
        """
        # Define p
        self.p = p

    def compute_distance(self, v1: pd.Series, v2: pd.Series) -> float:
        """
        Compute the Minkowski metric using the given order of p

        Parameters:
        -----------
        v1: pd.Series
            The first vector in the computation

        v1: pd.Series
            The second vector in the computation

        Returns:
        --------
        A float representing the distance between v1 and v2
        """
        dist = 0

        # Check if the two vectors are valid (as descried in the _validate_vectors method)
        if not self._validate_vectors(v1, v2):
            # If the they are invalid, raise an exception
            raise Exception(f"The shape of {v1} does not match the shape of {v2}.")

        # Iterate over all attributes and compute the difference between the two vectors
        for attribute in v1.index:
            dist += abs(v1[attribute] - v2[attribute]) ** self.p

        # Return the resultant distance
        return dist
