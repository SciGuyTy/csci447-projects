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

    def predict(self, instance, k):
        neighbors_distance = self.find_k_nearest_neighbors(instance, k)
        if self.regression:
            distances = [i[0] for i in neighbors_distance]
            return self.regress(distances)
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

    def regress(self, distances):
        total = 0
        for d in distances:
            total += self.gaussian_kernel(d/self.sigma)
        return total / (self.h * len(distances))

    def vote(self, neighbors):
        df = pd.DataFrame(data=neighbors)
        mode = df[self.class_col].mode()
        if len(mode) > 1:
            print("TIE!")
            mode = mode.sample()
        return mode.iloc[0]

    def gaussian_kernel(self, u):
        return math.e ** ((u ** 2) / -2) / math.sqrt(2 * math.pi)

    def minkowski_metric(self, x, y, p=2):
        dist = 0
        for attr_label in self.attributes:
            dist += abs(float(x[attr_label]) - float(y[attr_label])) ** p
        return dist
