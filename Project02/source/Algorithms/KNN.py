from typing import Any, List, Tuple
import pandas as pd
import math
from heapq import heappop, heappush, heapify
from itertools import count
from source.Algorithms.KMeans import KMeans

from source.Algorithms.DistanceFunctions.DistanceFunction import DistanceFunction
from source.Algorithms.DistanceFunctions.Minkowski import Minkowski


class KNN:
    def __init__(
        self,
        training_data: pd.DataFrame,
        target_feature: str,
        cluster: bool = False,
        regression: bool = False,
        sigma: float = None,
        distance_function: DistanceFunction = Minkowski(),
    ):
        """
        K-Nearest Neighbor algorithm that uses the response value of the k
        nearest points to predict the response value of a novel point

        Parameters:
        -----------
        training_data: pd.DataFrame
            The training data

        target_feature: str
            The name of target response feature

        cluster: bool
            Whether KMeans should be implemented (defaults to False)

        regression: bool
            Whether the training data is regression data (defaults to False)

        sigma: float
            A hyperparameter for the Gaussian Threshold (Default to None)

        distance_function: DistanceFunction
            The function used to compute the distance between two points
            (Defaults to the Minkowski Metric)
        """
        self.features = training_data.columns.drop(target_feature)
        self.target_feature = target_feature
        self.training_data: pd.DataFrame = training_data
        self.cluster = cluster
        self.regression = regression
        self.classes = self.training_data[target_feature].unique()
        self.tiebreaker = count()

        self.sigma = sigma
<<<<<<< HEAD
        if regression and self.sigma is None:
=======

        # Raise an exception if regression data is provided but the
        # Gaussian Threshold is not provide
        if regression and (self.sigma is None):
>>>>>>> 6848fda0e6988895f60e24ecdbf1c3eff1a43b52
            raise ValueError

        self.distance_function = distance_function

    def predict(self, instance: pd.Series, k: int) -> Any:
        """
        Predict the response value for a novel data point

        Parameters:
        -----------
        k: int
            The number of nearest neighbors to consider when making
            the prediction

        Returns:
        --------
        The prediction for the response value of the novel instance
        """
        # Array to store nearest neighbors distance
        neighbors_distance = []

        if self.cluster:
            # If clustering is selected, apply KMeans clustering
            kmeans = KMeans(self.features, self.training_data, self.distance_function)
            centroids = kmeans.cluster(k)

            # Find the distance between the novel instance and the k clusters
            for centroid in centroids:
                distance = self.distance_function.compute_distance(
                    centroid[self.features], instance[self.features]
                )

                neighbors_distance.append((distance, instance))
        else:
            # If clustering is not selected, simply find the k nearest neighbors
            neighbors_distance = self.find_k_nearest_neighbors(instance, k)

        if self.regression:
            # Perform regression and return the response
            return self.regress(neighbors_distance)
        else:
            # Apply a vote of plurality and return the response
            neighbors = [i[1] for i in neighbors_distance]
            return self.vote(neighbors)

    def find_k_nearest_neighbors(
        self, instance: pd.Series, k: int
    ) -> List[Tuple[float, pd.Series]]:
        """
        Find the k nearest data points for some provided instance

        Parameters:
        -----------
        instance: pd.Series
            The instance which which to find neighbors for

        k: int
            The number of nearest neighbors to consider when making
            the prediction

        Returns:
        --------
        A list of the k nearest neighbors and their respective distances
        """
        # Define a bottom-up heap to store neighbor data
        heap = []
        heapify(heap)

        # For each data point in the training data...
        for i in range(len(self.training_data.index)):
            neighbor = self.training_data.iloc[i]

            # Compute the distance from the instance point
            distance = -1 * self.distance_function.compute_distance(
                neighbor[self.features], instance[self.features]
            )

            # If the distance is less than the largest distance on the heap,
            # pop the largest distance and push this new smaller distance
            if len(heap) < k or abs(distance) < abs(heap[0][0]):
                # Push the distance, a tiebreaker, and the neighbor onto the heap
                heappush(heap, (distance, next(self.tiebreaker), neighbor))
            if len(heap) > k:
                heappop(heap)

        # Flip the distances and remove tiebreaker
        return [(-i[0], i[2]) for i in heap]

    def regress(self, neighbors_distances: List[Tuple[float, pd.Series]]) -> float:
        """
        Perform regression on a set of neighbors for the response value

        Parameters:
        -----------
        neighbors_distances: List[Tuple[float, pd.Series]]
            The neighbors to use for regression

        Returns:
        --------
        A float representing the predicted response value
        """
        total = 0
        weighted_sum = 0

        # For each neighbor, compute apply the gaussian kernel
        for d, neighbor in neighbors_distances:
            kernel_distance = self.gaussian_kernel(d, 1 / self.sigma)
            total += kernel_distance
            weighted_sum += kernel_distance * neighbor[self.target_feature]

        if total == 0:
            return 0
        else:
            return weighted_sum / total

    def vote(self, neighbors: List[pd.Series]) -> Any:
        """
        Perform a vote of plurality

        Parameters:
        -----------
            A list of neighbors with which to find the mode response
            value for

        Returns: 
        --------
        The mode response value
        """
        # Combine neighbors into a DataFrame
        df = pd.DataFrame(data=neighbors)

        # Find the mode of the dataFrame for the response value
        mode = df[self.target_feature].mode()

        # Return the most common response value
        if len(mode) > 1:
            # If there are ties, sample one at random
            mode = mode.sample()
        return mode.iloc[0]

    @staticmethod
    def gaussian_kernel(distance: float, gamma: float) -> float:
        """
        Representation of the Gaussian Kernel function

        Parameters:
        -----------
        distance: float
            The distance to apply the kernel to

        gamma: float
            The Gaussian Threshold

        Returns: 
        --------
        The weighted distance
        """
        return math.e ** (-gamma * (distance**2))
