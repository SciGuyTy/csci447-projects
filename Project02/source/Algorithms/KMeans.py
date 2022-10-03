from typing import List
from source.Algorithms.DistanceFunctions.Minkowski import Minkowski
from source.Algorithms.DistanceFunctions.DistanceFunction import DistanceFunction

import math
import pandas as pd


class KMeans:
    def __init__(
        self,
        features: List[str],
        data: pd.DataFrame,
        distance_function: DistanceFunction = Minkowski(),
    ):
        """
        K-Means clustering used to cluster data

        Parameters:
        -----------
        features: List[str]
            A list of strings that represent the training features

        data: pd.DataFrame
            The training data

        distance_function: DistanceFunction
            The distance function to use when computing the distance
            between a centroid and a sample (defaults to the Minkoski metric
            with p=2)
        """
        self.features = features
        self.data = data
        self.distance_function = distance_function

    def _initialize_clusters(self) -> None:
        """
        Handles initialization the k cluster objects, containing a reference to
        the cluster centroid as well as the samples which belong to the cluster
        """
        # Define an array to hold cluster objects
        clusters = []

        for cluster_id in range(self.k):
            # Select a random sample from the dataset to act as an initial
            # centroid location
            centroid = self.data.sample().iloc[0][self.features]

            # Define an empty DataFrame with which to hold samples belonging
            # to the cluster
            samples = pd.DataFrame(columns=self.data.columns)

            # Define the cluster object
            clusters.append({"centroid": centroid, "samples": samples})

        self.clusters = clusters

    def _update_samples(self) -> None:
        """
        Handles updating the samples that belong to each cluster
        """
        # Locate the nearest cluster centroid for each sample in the dataset
        for _, row in self.data.iterrows():
            # Rows current shortest distance to the 'current_centroid'
            current_dist = math.inf
            current_centroid = None

            for cluster_id, cluster in enumerate(self.clusters):
                # Compute the distance between the sample and the current centroid
                dist = self.distance_function.compute_distance(
                    row[self.features], cluster["centroid"]
                )

                # If the sample is closer to the given centroid than the previous
                # centroid, update the 'current_centroid' to reflect
                if dist < current_dist:
                    current_dist = dist
                    current_centroid = cluster_id

            # Reference to samples DataFrame for the current cluster
            cluster_samples = self.clusters[current_centroid]["samples"]

            # Updated samples DataFrame containing the 'clustered' row
            updated_cluster_samples = pd.concat(
                [cluster_samples, row.to_frame().T], ignore_index=True
            )

            # Update the samples DataFrame for the current cluster
            self.clusters[current_centroid]["samples"] = updated_cluster_samples

    def _update_centroids(self) -> None:
        """
        Handles updating the centroid of a cluster based on the new mean
        after updating the samples
        """
        # Move the centroid for each cluster to the new center
        for index_id, cluster in enumerate(self.clusters):
            # Compute the mean vector for the given cluster
            mean_vector = cluster["samples"].mean()[self.features]

            # Update the centroid to the new center of the cluster
            self.clusters[index_id]["centroid"] = mean_vector

    def cluster(self, k: int) -> None:
        """
        Cluster data into k clusters using the k-means algorithm

        Parameters:
        -----------
        k: int
            Number of clusters to fit
        """
        self.k = k

        # Initialize the clusters
        self._initialize_clusters()

        # Reference to the current clusters (used to keep track of centroid movement)
        current_clusters = None

        while True:
            # Update the samples and move the centroids based on the new cluster mean
            self._update_samples()
            self._update_centroids()

            # If no centroids change, stop the loop
            if self.clusters == current_clusters:
                break

            # Update the 'previous' cluster data
            current_clusters = self.clusters

        return [cluster["centroid"] for cluster in current_clusters]
