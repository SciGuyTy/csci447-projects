import random
from typing import List

from Project04.Algorithms.Crossover.Crossover import Crossover


class UniformCrossover(Crossover):
    def __init__(self, probability_of_cross: float = 0.8):
        super().__init__(probability_of_cross)

    def cross(self, parent_one: List[float], parent_two: List[float]) -> List[list[float]]:
        """
        Perform uniform crossover using two chromosomes

        Parameters
        ----------
        parent_one: List[float]
            The first parent to cross

        parent_two: List[float]
            The second parent to cross

        Returns
        -------
        A list of the children (crossed) chromosomes
        """

        child_one = parent_one.copy()
        child_two = parent_two.copy()

        for index in range(len(child_one)):
            # Only perform a crossover based on the provided probability of a cross occurring
            if random.random() > (1 - self.probability_of_cross):
                # Swap the two values
                child_one_value = child_one[index]
                child_one[index] = child_two[index]
                child_two[index] = child_one_value

        return [child_one, child_two]
