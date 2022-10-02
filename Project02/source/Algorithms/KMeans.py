import math
from threading import current_thread
from tkinter import W
from typing import List
from unicodedata import name
import pandas as pd
from source.Algorithms.DistanceFunctions.Minkowski import Minkowski
from source.Algorithms.DistanceFunctions.DistanceFunction import DistanceFunction


class KMeans:
    def __init__(
        self,
        attributes: List[str],
        training_data: pd.DataFrame,
        class_col: str,
        k: int,
        distance_function: DistanceFunction = Minkowski(),
    ) -> None:
        self.attributes = attributes
        self.training_data = training_data
        self.class_col = class_col
        self.k = k
        self.distance_function = distance_function

    def _initialize_clusters(self):
        self.clusters = self.training_data.groupby(self.class_col).sample(1)

    def _update_cluster(self):
        for row_index, row in self.training_data.iterrows():
            current_dist = math.inf
            current_classification = None

            for cluster in self.clusters.iterrows():
                cluster = cluster[1]
                dist = self.distance_function.compute_distance(
                    row[self.attributes], cluster[self.attributes]
                )

                if dist < current_dist:
                    current_dist = dist
                    current_classification = cluster["class"]

            self.training_data.at[row_index, "class"] = current_classification

    def _move_centroid(self):
        for index, cluster in self.clusters.iterrows():
            self.clusters.loc[index, self.attributes] = self.training_data[
                self.training_data["class"] == cluster["class"]
            ].mean()[self.attributes]

    def _cluster(self):
        self._initialize_clusters()

        current_clusters = None

        while True:
            self._update_cluster()
            self._move_centroid()
            print(self.clusters)

            if self.clusters.equals(current_clusters):
                break

            current_clusters = self.clusters.copy()

    def predict(self, instance):

        # print(self.training_data[self.training_data["class"] == True].mean())

        self._cluster()

        results = dict.fromkeys(self.training_data["class"].unique())

        for _, cluster in self.clusters.iterrows():
            dist = self.distance_function.compute_distance(
                instance[self.attributes], cluster[self.attributes]
            )

            results[cluster["class"]] = dist

        print(min(results))
