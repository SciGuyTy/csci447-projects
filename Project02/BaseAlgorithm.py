from abc import ABC, abstractmethod
import pandas as pd
import math
from heapq import heappop, heappush, heapify

class BaseAlgorithm(ABC):

    def __init__(self, attributes, class_col: str, training_data, k, h, sigma, regression=False):
        self.attributes = attributes
        self.class_col = class_col
        self.training_data = training_data
        self.regression = regression
        self.classes = self.training_data[self.class_col].unique()

        self.h = h
        self.sigma = sigma
        self.k = k

    def predict(self, instance, k):
        pass

    def find_k_nearest_neighbors(self, instance, k):
        heap = []
        heapify(heap)
        for neighbor in self.training_data:
            distance = -1 * self.minkowski_metric(neighbor, instance) # Multiple by negative one to use Max Heap
            if len(heap) < k:
                heappush(heap, (distance, neighbor))
            elif abs(distance) < abs(heap[0]):
                heappop(heap)
                heappush(heap, (distance, neighbor))
        return heap # Flip distance TODO

    def regress(self, neighbors, instance):
        total = 0
        for t in neighbors:
            pass
            # TODO
        
    def vote(self, neighbors):
        return neighbors[self.class_col].mode()

    def gaussian_kernel(self, u):
        return math.e**((u**2)/-2) / math.sqrt(2*math.pi)

    def minkowski_metric(self, x, y, p=2):
        dist = 0
        for attr_label in self.attributes:
            dist += abs(x[attr_label] - y[attr_label])**p
        return dist
