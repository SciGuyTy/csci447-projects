import pandas as pd


class DistanceFunction:
    def __init__(self) -> None:
        """
        "Interface" for distance functions
        """
        pass

    def _preprocess(self) -> None:
        """
        Handle any preprocessing of the vectors required by the underlying distance
        function
        """
        pass

    def _validate_vectors(self, v1: pd.Series, v2: pd.Series) -> bool:
        """
        This method is used to validate the two vectors. By default, the shape of the
        two vectors is compared to ensure they share the same attributes

        Parameters:
        ----------
        v1: pd.Series
            The first vector to validate

        v1: pd.Series
            The second vector to validate

        Returns:
        --------
        A boolean representing the validity of both v1 and v2

        """
        return v1.shape == v2.shape

    def compute_distance(self, v1: pd.Series, v2: pd.Series) -> float:
        """
        Compute the distance between two vectors

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
        self._validate_vectors()
        self._preprocess()
