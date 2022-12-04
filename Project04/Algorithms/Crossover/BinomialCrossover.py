import random
from typing import List

from Project04.Algorithms.Crossover.Crossover import Crossover


class BinomialCrossover(Crossover):
    def __init__(self, probability_of_cross: float = 0.8):
        super().__init__(probability_of_cross)

    def cross(self, parent_one: List[float], parent_two: List[float]) -> list[float]:
        """
        Perform binomial crossover using two chromosomes

        Parameters
        ----------
        parent_one: List[float]
            The target chromosome with which to modify

        parent_two: List[float]
            The trial chromosome

        Returns
        -------
        A list of floats representing the crossed chromosome
        """

        # Resultant chromosome
        crossed_chromosome = []

        # Iterate through each dimension in the target chromosome and...
        for i in range(len(parent_one)):
            # Randomly replace the gene at this location with that of the trial chromosome
            if random.random() > (1 - self.probability_of_cross):
                crossed_chromosome.append(parent_two[i])
            else:
                crossed_chromosome.append(parent_one[i])

        # Return the crossed chromosome
        return crossed_chromosome
