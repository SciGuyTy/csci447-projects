import pandas as pd
import math
from heapq import heappop, heappush, heapify
from itertools import count

from source.Algorithms.DistanceFunctions.DistanceFunction import DistanceFunction
from source.Algorithms.DistanceFunctions.Minkowski import Minkowski

class KNN:
    def __init__(
        self,
        training_data: pd.DataFrame,
        target_feature: str,
        regression=False,
        sigma=None,
        distance_function: DistanceFunction = Minkowski(),
    ):
        self.features = training_data.columns.drop(target_feature)
        self.target_feature = target_feature
        self.training_data: pd.DataFrame = training_data
        self.regression = regression
        self.classes = self.training_data[target_feature].unique()
        self.tiebreaker = count()

        self.sigma = sigma
        if regression and (self.sigma is None):
            raise ValueError

        self.distance_function = distance_function

    def predict(self, instance, k):
        neighbors_distance = self.find_k_nearest_neighbors(instance, k)

        if self.regression:
            return self.regress(neighbors_distance)
        else:
            neighbors = [i[1] for i in neighbors_distance]
            return self.vote(neighbors)

    def find_k_nearest_neighbors(self, instance, k):
        heap = []
        heapify(heap)
        for i in range(len(self.training_data.index)):
            neighbor = self.training_data.iloc[i]
            distance = -1 * self.distance_function.compute_distance(
                neighbor[self.features], instance[self.features]
            )
            if len(heap) < k or abs(distance) < abs(heap[0][0]):
                # Push the distance, a tiebreaker, and the neighbor onto the heap
                heappush(heap, (distance, next(self.tiebreaker), neighbor))
            if len(heap) > k:
                heappop(heap)
        return [(-i[0], i[2]) for i in heap]  # Flip distance and remove tiebreaker

    def regress(self, neighbors_distances):
        total = 0
        weighted_sum = 0
        for d, neighbor in neighbors_distances:
            kernel_distance = self.gaussian_kernel(d, 1/self.sigma)
            total += kernel_distance
            weighted_sum += kernel_distance * neighbor[self.target_feature]

        if weighted_sum == 0:
            return 0
        else:
            return weighted_sum / total

    def vote(self, neighbors):
        df = pd.DataFrame(data=neighbors)
        mode = df[self.target_feature].mode()
        if len(mode) > 1:
            mode = mode.sample()
        return mode.iloc[0]

    @staticmethod
    def gaussian_kernel(distance, gamma):
        return math.e ** (-gamma * (distance ** 2))