from abc import ABC, abstractmethod
import pandas as pd
import math
from heapq import heappop, heappush, heapify
from itertools import count


class BaseAlgorithm(ABC):

    def __init__(self, attributes, class_col: str, training_data: pd.DataFrame, regression=False, h=None, sigma=None):
        self.attributes = attributes
        self.class_col = class_col
        self.training_data: pd.DataFrame = training_data
        self.regression = regression
        self.classes = self.training_data[self.class_col].unique()
        self.tiebreaker = count()

        self.h = h
        self.sigma = sigma
        if regression and (self.sigma is None or self.h is None):
            raise ValueError

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
            distance = -1 * self.minkowski_metric(neighbor, instance)  # Multiply by negative one to use Max Heap
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
            kernel_distance = self.gaussian_kernel(d/self.sigma)
            total += kernel_distance
            weighted_sum += kernel_distance * neighbor[self.class_col]
        return weighted_sum / total

    def vote(self, neighbors):
        df = pd.DataFrame(data=neighbors)
        mode = df[self.class_col].mode()
        if len(mode) > 1:
            print("TIE!")
            mode = mode.sample()
        return mode.iloc[0]

    @staticmethod
    def gaussian_kernel(u):
        return math.e ** ((u ** 2) / -2) / math.sqrt(2 * math.pi)

    def minkowski_metric(self, x, y, p=2):
        dist = 0
        for attr_label in self.attributes:
            dist += abs(float(x[attr_label]) - float(y[attr_label])) ** p
        return dist
